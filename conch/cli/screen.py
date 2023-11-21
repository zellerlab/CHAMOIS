import argparse
import pathlib
import time
import typing
from typing import List, Iterable, Set, Optional

import anndata
import numpy
import pandas
import rich.table
import scipy.stats
from rich.console import Console
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from scipy.spatial.distance import cdist

from .._meta import requires
from ..predictor import ChemicalOntologyPredictor
from ..classyfire import query_classyfire, get_results, extract_classification, binarize_classification
from ._common import load_model
from .render import build_tree
from ._parser import (
    configure_group_search_input,
    configure_group_search_parameters,
    configure_group_search_output,
)

if typing.TYPE_CHECKING:
    from rdkit.Chem import Mol

@requires("rdkit.Chem")
@requires("rdkit.RDLogger")
def _parse_molecule(text: str) -> "Mol":
    rdkit.RDLogger.DisableLog('rdApp.error')
    for parse in (rdkit.Chem.MolFromInchi, rdkit.Chem.MolFromSmiles):
        mol = parse(text)
        if mol is not None:
            return mol
    raise ValueError(f"Could not parse {molecule!r} molecule")


def configure_parser(parser: argparse.ArgumentParser):
    params_input = configure_group_search_input(parser)
    params_input.add_argument(
        "-q",
        "--query",
        action="append",
        required=True,
        dest="queries",
        type=_parse_molecule,
        help="The compounds to search in the predictions.",
    )
    configure_group_search_parameters(parser)
    configure_group_search_output(parser)
    parser.set_defaults(run=run)


def load_predictions(path: pathlib.Path, predictor: ChemicalOntologyPredictor, console: Console) -> anndata.AnnData:
    console.print(f"[bold blue]{'Loading':>12}[/] probability predictions from {str(path)!r}")
    probas = anndata.read(path)
    probas = probas[:, predictor.classes_.index]
    classes = predictor.propagate(probas.X > 0.5)
    return probas, anndata.AnnData(X=classes, obs=probas.obs, var=probas.var, dtype=bool)


@requires("rdkit.Chem")
def build_results(
    queries: List["Mol"],
    classes: anndata.AnnData,
    distances: numpy.ndarray,
    ranks: numpy.ndarray,
    max_rank: int,
) -> pandas.DataFrame:
    rows = []
    for i, query in enumerate(queries):
        for j in ranks[i].argsort():
            if ranks[i, j] > max_rank:
                break
            rows.append([
                rdkit.Chem.MolToInchiKey(query),
                rdkit.Chem.MolToInchi(query),
                rdkit.Chem.MolToSmiles(query),
                ranks[i, j],
                classes.obs_names[j],
                *classes.obs.iloc[j],
                distances[i, j],
            ])
    return pandas.DataFrame(
        rows,
        columns=[
            "inchikey",
            "inchi",
            "smiles",
            "rank",
            "bgc_id",
            *classes.obs.columns,
            "distance"
        ]
    )


def build_table(results: pandas.DataFrame) -> rich.table.Table:
    table = rich.table.Table("Compound", "BGC", "Distance")
    for compound, rows in results[results["rank"] == 1].groupby("compound", sort=False):
        for i, row in enumerate(rows.itertuples()):
            table.add_row(
                row.compound if i == 0 else "",
                rich.text.Text(row.bgc_id, style="repr.tag_name"),
                rich.text.Text(format(row.distance, ".5f"), style="repr.number"),
                end_section=i==len(rows)-1,
            )
    return table


@requires("rdkit.Chem")
@requires("rdkit.RDLogger")
def run(args: argparse.Namespace, console: Console) -> int:
    rdkit.RDLogger.DisableLog('rdApp.warning')
    rdkit.RDLogger.DisableLog('rdApp.info')

    predictor = load_model(args.model, console)
    probas, classes = load_predictions(args.input, predictor, console)

    # send classification job to classyfire
    console.print(f"[bold blue]{'Sending':>12}[/] {len(args.queries)} queries to ClassyFire for classification")
    queries = [rdkit.Chem.inchi.MolToInchi(mol) for mol in args.queries]
    response = query_classyfire(queries)
    if "id" in response:
        query_id = response["id"]
    else:
        raise RuntimeError("Failed to submit queries to ClassyFire: ")

    # retrieve classification results
    console.print(f"[bold blue]{'Waiting':>12}[/] for ClassyFire to process query {query_id}")
    while True:
        time.sleep(5.0)
        results = get_results(query_id)
        if results['classification_status'] == 'Done':
            break
        elif results['classification_status'] not in {"In progress", "In Queue"}:
            console.print(f"[bold red]{'Failed':>12}[/] to get classification ({results['classification_status']!r})")
            return 1

    # Report failed classification
    for entity in results["invalid_entities"]:
        console.print(f"[bold red]{'Failed':>12}[/] to classify {entity['structure']!r}")

    # retrieve classification results
    classifications = {}
    for i in range(results['number_of_pages']):
        for entity in results['entities']:
            inchikey = entity["inchikey"].split("=")[-1]
            classifications[inchikey] = extract_classification(entity)
        if i+1 < results['number_of_pages']:
            results = get_results(query_id, page=i+1)

    # binarize classifications
    compounds = numpy.zeros((len(args.queries), len(predictor.classes_)))
    for i, query in enumerate(args.queries):
        inchikey = rdkit.Chem.inchi.MolToInchiKey(query)
        leaves = classifications[inchikey]
        compounds[i] = binarize_classification(
            predictor.classes_,
            predictor.ontology.incidence_matrix,
            leaves
        )

    def metric(x: numpy.ndarray, y: numpy.ndarray) -> float:
        i = numpy.where(x)[0]
        j = numpy.where(y)[0]
        return 1.0 - predictor.ontology.similarity(i, j)

    # compute distance
    console.print(f"[bold blue]{'Computing':>12}[/] distances to predictions")
    distances = cdist(compounds, classes.X, metric=metric)
    ranks = scipy.stats.rankdata(distances, method="dense", axis=1)

    # show most likely BGC for input compound
    if args.render:
        table = Table.grid()
        table.add_column(no_wrap=True)
        table.add_column(no_wrap=True)
        for i, query in enumerate(args.queries):
            j = ranks[i].argmin()
            tree_bgc = build_tree(predictor, probas.X[j])
            tree_query = build_tree(predictor, compounds[i])
            inchikey = rdkit.Chem.inchi.MolToInchiKey(query)
            table.add_row(
                Panel(tree_query, title=f"[bold purple]{inchikey}[/]"),
                Panel(tree_bgc, title=f"[bold purple]{probas.obs.index[j]}[/] (d=[bold cyan]{distances[i, j]:.5f}[/])"),
            )
        console.print(table)

    # save results
    if args.output:
        results = build_results(args.queries, classes, distances, ranks, max_rank=args.rank)
        console.print(f"[bold blue]{'Saving':>12}[/] search results to {str(args.output)!r}")
        results.to_csv(args.output, sep="\t", index=False)
