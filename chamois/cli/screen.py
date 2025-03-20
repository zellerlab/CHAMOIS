import abc
import argparse
import contextlib
import dataclasses
import errno
import pathlib
import time
import json
import re
import urllib.request
import typing
from typing import List, Iterable, Set, Optional

import numpy
import rich.table
from rich.console import Console
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table

from .._meta import requires
from ..predictor import ChemicalOntologyPredictor
from ..classyfire import Client, binarize_classification
from ._common import load_model
from .render import build_tree
from .search import probjaccard, probjaccard_cdist
from ._parser import (
    configure_group_search_input,
    configure_group_search_parameters,
    configure_group_search_output,
)

if typing.TYPE_CHECKING:
    from rdkit.Chem import Mol
    from anndata import AnnData
    from pandas import DataFrame


@dataclasses.dataclass
class Query(abc.ABC):
    pass


@dataclasses.dataclass
class MolQuery(Query):
    mol: "Mol"

    @classmethod
    @requires("rdkit.Chem")
    @requires("rdkit.RDLogger")
    def parse(cls, text: str) -> "MolQuery":
        rdkit.RDLogger.DisableLog('rdApp.error')
        for parse in (rdkit.Chem.MolFromInchi, rdkit.Chem.MolFromSmiles):
            mol = parse(text)
            if mol is not None:
                return cls(mol)
        raise ValueError(text)

    @property
    @requires("rdkit.Chem")
    def inchikey(self):
        return rdkit.Chem.MolToInchiKey(self.mol)

    @property
    @requires("rdkit.Chem")
    def inchi(self):
        return rdkit.Chem.MolToInchi(self.mol)

    @property
    @requires("rdkit.Chem")
    def smiles(self):
        return rdkit.Chem.MolToSmiles(self.mol)


_INCHIKEY_RX = re.compile(r"[A-Z]{14}-[A-Z]{10}-[A-Z]")

@dataclasses.dataclass
class InchiKeyQuery(Query):
    inchikey: str

    @classmethod
    def parse(cls, text: str) -> "InchiKeyQuery":
        if not _INCHIKEY_RX.match(text):
            raise ValueError(text)
        return cls(text)

    @property
    def inchi(self):
        return None

    @property
    def smiles(self):
        return None


def parse_query(text: str) -> Query:
    with contextlib.suppress(ValueError):
        return InchiKeyQuery.parse(text)
    with contextlib.suppress(ValueError):
        return MolQuery.parse(text)
    raise ValueError(f"Could not parse {text!r} molecule")


def configure_parser(parser: argparse.ArgumentParser):
    params_input = configure_group_search_input(parser)
    params_input.add_argument(
        "-q",
        "--query",
        action="append",
        required=True,
        dest="queries",
        help="The compounds to search in the predictions, as a SMILES, InChi, or InChiKey.",
    )
    # configure_group_search_parameters(parser)
    configure_group_search_output(parser)
    parser.set_defaults(run=run)


def load_predictions(path: pathlib.Path, predictor: ChemicalOntologyPredictor, console: Console) -> "AnnData":
    import anndata

    console.print(f"[bold blue]{'Loading':>12}[/] probability predictions from {str(path)!r}")
    probas = anndata.read_h5ad(path)
    probas = probas[:, predictor.classes_.index]
    classes = predictor.propagate(probas.X > 0.5)
    return probas, anndata.AnnData(X=classes, obs=probas.obs, var=probas.var)


@requires("pandas")
def build_results(
    queries: List[Query],
    classes: "AnnData",
    distances: numpy.ndarray,
    ranks: numpy.ndarray,
    max_rank: int,
) -> "DataFrame":
    rows = []
    for i, query in enumerate(queries):
        for j in ranks[i].argsort():
            if ranks[i, j] > max_rank:
                break
            rows.append([
                query.inchikey,
                query.inchi,
                query.smiles,
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


def build_table(results: "DataFrame") -> rich.table.Table:
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


@requires("scipy.stats")
def run(args: argparse.Namespace, console: Console) -> int:
    # silence the RDKit logger if installed
    try:
        import rdkit.RDLogger
        rdkit.RDLogger.DisableLog('rdApp.warning')
        rdkit.RDLogger.DisableLog('rdApp.info')
    except ImportError:
        RDLogger = None

    # parse queries
    queries = []
    for query in args.queries:
        try:
            queries.append(parse_query(query))
        except ValueError:
            console.print(f"[bold red]{'Failed':>12}[/] to parse query: {query!r}")
            return errno.EINVAL

    # load predictor
    predictor = load_model(args.model, console)
    probas, classes = load_predictions(args.input, predictor, console)

    # get query classification from Classyfire
    classifications = {}
    classyfire = Client()
    for query in queries:
        inchikey = query.inchikey
        console.print(f"[bold blue]{'Retrieving':>12}[/] {len(queries)} ClassyFire results for [bold cyan]{inchikey}[/]")
        classifications[inchikey] = classyfire.fetch(inchikey)

    # binarize classifications
    compounds = numpy.zeros((len(queries), len(predictor.classes_)))
    for i, query in enumerate(queries):
        inchikey = query.inchikey
        leaves = [t.id for t in classifications[inchikey].terms]
        compounds[i] = binarize_classification(
            predictor.classes_,
            predictor.ontology.adjacency_matrix,
            leaves
        )
    
    # compute distances
    console.print(f"[bold blue]{'Computing':>12}[/] distances to predictions")
    distances = probjaccard_cdist(compounds, probas.X)
    ranks = scipy.stats.rankdata(distances, method="dense", axis=1)

    # show most likely BGC for input compound
    if args.render:
        table = Table.grid()
        table.add_column(no_wrap=True)
        table.add_column(no_wrap=True)
        for i, query in enumerate(queries):
            j = ranks[i].argmin()
            tree_bgc = build_tree(predictor, probas.X[j])
            tree_query = build_tree(predictor, compounds[i])
            inchikey = query.inchikey
            table.add_row(
                Panel(tree_query, title=f"[bold purple]{inchikey}[/]"),
                Panel(tree_bgc, title=f"[bold purple]{probas.obs.index[j]}[/] (d=[bold cyan]{distances[i, j]:.5f}[/])"),
            )
        console.print(table)

    # save results
    if args.output:
        results = build_results(queries, classes, distances, ranks, max_rank=args.rank)
        console.print(f"[bold blue]{'Saving':>12}[/] search results to {str(args.output)!r}")
        results.to_csv(args.output, sep="\t", index=False)

    return 0