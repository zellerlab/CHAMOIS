import argparse
import pathlib
from typing import List, Iterable, Set, Optional

import anndata
import numpy
import pandas
import rich.table
import scipy.stats
from rich.console import Console
from rich.columns import Columns
from rich.panel import Panel
from scipy.spatial.distance import cdist

from ..predictor import ChemicalHierarchyPredictor
from ..classyfire import query_classyfire, extract_classification, binarize_classification
from ._common import load_model
from .render import build_tree


def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=pathlib.Path,
        help="The chemical class probabilities predicted for BGCs."
    )
    parser.add_argument(
        "-k", 
        "--inchikey",
        action="append",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help="The path where to write the catalog search results in TSV format."
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        type=pathlib.Path,
        help="The path to an alternative model for predicting classes."
    )
    parser.add_argument(
        "-d",
        "--distance",
        default="hamming",
        help="The metric to use for comparing classes fingerprints.",
        choices={"hamming", "jaccard"},
    )
    parser.add_argument(
        "--rank",
        default=10,
        type=int,
        help="The maximum search rank to record in the table output.",
    )
    parser.set_defaults(run=run)


def load_predictions(path: pathlib.Path, predictor: ChemicalHierarchyPredictor, console: Console) -> anndata.AnnData:
    console.print(f"[bold blue]{'Loading':>12}[/] probability predictions from {str(path)!r}")
    probas = anndata.read(path)
    probas = probas[:, predictor.classes_.index]
    classes = predictor.propagate(probas.X > 0.5)
    return probas, anndata.AnnData(X=classes, obs=probas.obs, var=probas.var, dtype=bool)


def build_results(
    queries: List[str], 
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
                query,
                ranks[i, j],
                classes.obs.index[j], 
                distances[i, j],
            ])
    return pandas.DataFrame(
        rows, 
        columns=["compound", "rank", "bgc_id", "distance"]
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


def run(args: argparse.Namespace, console: Console) -> int:
    predictor = load_model(args.model, console)
    probas, classes = load_predictions(args.input, predictor, console)

    # get classification for compounds
    console.print(f"[bold blue]{'Querying':>12}[/] ClassyFire for compound classifications")
    compounds = numpy.zeros((len(args.inchikey), len(predictor.classes_)))
    for i, key in enumerate(args.inchikey):
        response = query_classyfire(key, wait=5.0)
        leaves = extract_classification(response)
        compounds[i] = binarize_classification(predictor.classes_, predictor.hierarchy, leaves)

    # compute distance
    console.print(f"[bold blue]{'Computing':>12}[/] distances to predictions")
    distances = cdist(compounds, classes.X, metric=args.distance)
    ranks = scipy.stats.rankdata(distances, method="dense", axis=1)

    # show most likely BGC for input compound
    for i, key in enumerate(args.inchikey):
        j = ranks[i].argmin()
        tree_bgc = build_tree(predictor, probas.X[j])
        tree_query = build_tree(predictor, compounds[i])
        console.print(
            Columns([
                Panel(tree_query, title=f"[bold purple]{key}[/]"), 
                Panel(tree_bgc, title=f"[bold purple]{probas.obs.index[j]}[/] (d=[bold cyan]{distances[i, j]:.5f}[/])")
            ])
        )


    # # save results
    # results = build_results(args.inchikey, classes, distances, ranks, max_rank=args.rank)
    # if args.output:
    #     console.print(f"[bold blue]{'Saving':>12}[/] search results to {str(args.output)!r}")
    #     results.to_csv(args.output, sep="\t", index=False)

    # # display output
    # table = build_table(results)
    # console.print(table)