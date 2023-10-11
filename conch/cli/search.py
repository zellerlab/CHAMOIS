import argparse
import pathlib
from typing import List, Iterable, Set, Optional

import anndata
import numpy
import pandas
import rich.table
import scipy.stats
from rich.console import Console
from scipy.spatial.distance import cdist

from ._common import load_model
from ..predictor import ChemicalOntologyPredictor


def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=pathlib.Path,
        help="The chemical class probabilities predicted for BGCs."
    )
    parser.add_argument(
        "-c",
        "--catalog",
        required=True,
        type=pathlib.Path,
        help="The path to the compound class catalog to compare predictions to."
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


def load_predictions(path: pathlib.Path, predictor: ChemicalOntologyPredictor, console: Console) -> anndata.AnnData:
    console.print(f"[bold blue]{'Loading':>12}[/] probability predictions from {str(path)!r}")
    probas = anndata.read(path)
    probas = probas[:, predictor.classes_.index]
    classes = predictor.propagate(probas.X > 0.5)
    return anndata.AnnData(X=classes, obs=probas.obs, var=probas.var, dtype=bool)


def load_catalog(path: pathlib.Path, console: Console) -> anndata.AnnData:
    console.print(f"[bold blue]{'Loading':>12}[/] compound catalog from {str(path)!r}")
    catalog = anndata.read(path)
    return catalog


def build_results(
    classes: anndata.AnnData,
    catalog: anndata.AnnData,
    distances: numpy.ndarray,
    ranks: numpy.ndarray,
    max_rank: int,
) -> pandas.DataFrame:
    rows = []
    for i, name in enumerate(classes.obs_names):
        for j in ranks[i].argsort():
            if ranks[i, j] > max_rank:
                break
            rows.append([
                name,
                ranks[i, j],
                catalog.obs.index[j],
                catalog.obs.compound[j],
                distances[i, j],
            ])
    return pandas.DataFrame(
        rows,
        columns=["bgc_id", "rank", "index", "compound", "distance"]
    )


def build_table(results: pandas.DataFrame) -> rich.table.Table:
    table = rich.table.Table("BGC", "Index", "Compound", "Distance")
    for bgc_id, rows in results[results["rank"] == 1].groupby("bgc_id", sort=False):
        for i, row in enumerate(rows.itertuples()):
            table.add_row(
                row.bgc_id if i == 0 else "",
                row.index,
                rich.text.Text(row.compound, style="repr.tag_name"),
                rich.text.Text(format(row.distance, ".5f"), style="repr.number"),
                end_section=i==len(rows)-1,
            )
    return table


def run(args: argparse.Namespace, console: Console) -> int:
    predictor = load_model(args.model, console)
    classes = load_predictions(args.input, predictor, console)

    # load catalog
    catalog = load_catalog(args.catalog, console)[:, classes.var.index]

    def metric(x: numpy.ndarray, y: numpy.ndarray) -> float:
        i = numpy.where(x)[0]
        j = numpy.where(y)[0]
        return 1.0 - predictor.ontology.similarity(i, j)

    # compute distance
    console.print(f"[bold blue]{'Computing':>12}[/] pairwise distances and ranks")
    distances = cdist(classes.X, catalog.X.toarray(), metric=metric)
    ranks = scipy.stats.rankdata(distances, method="dense", axis=1)

    # save results
    results = build_results(classes, catalog, distances, ranks, max_rank=args.rank)
    if args.output:
        console.print(f"[bold blue]{'Saving':>12}[/] search results to {str(args.output)!r}")
        results.to_csv(args.output, sep="\t", index=False)

    # display output
    table = build_table(results)
    console.print(table)