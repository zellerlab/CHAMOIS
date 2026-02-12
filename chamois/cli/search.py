import argparse
import pathlib
import typing
from typing import List, Iterable, Set, Optional

import numpy
import rich.table
from rich.console import Console

from .._meta import requires
from ..predictor import ChemicalOntologyPredictor
from ._common import load_model, record_metadata
from ._parser import (
    configure_group_search_input,
    configure_group_search_parameters,
    configure_group_search_output,
)

if typing.TYPE_CHECKING:
    from anndata import AnnData
    from pandas import DataFrame

def configure_parser(parser: argparse.ArgumentParser):
    params_input = configure_group_search_input(parser)
    params_input.add_argument(
        "-c",
        "--catalog",
        required=True,
        type=pathlib.Path,
        help="The path to the compound class catalog to compare predictions to."
    )
    # configure_group_search_parameters(parser)
    configure_group_search_output(parser)
    parser.set_defaults(run=run)


@requires("anndata")
def load_predictions(path: pathlib.Path, predictor: ChemicalOntologyPredictor, console: Console) -> "AnnData":
    console.print(f"[bold blue]{'Loading':>12}[/] probability predictions from {str(path)!r}")
    probas = anndata.read_h5ad(path)
    probas = probas[:, predictor.classes_.index]
    classes = predictor.propagate(probas.X > 0.5)
    return probas, anndata.AnnData(X=classes, obs=probas.obs, var=probas.var)


@requires("anndata")
def load_catalog(path: pathlib.Path, predictor: ChemicalOntologyPredictor, console: Console) -> "AnnData":
    console.print(f"[bold blue]{'Loading':>12}[/] compound catalog from {str(path)!r}")
    catalog = anndata.read_h5ad(path)
    return catalog


@requires("pandas")
def build_results(
    classes: "AnnData",
    catalog: "AnnData",
    distances: numpy.ndarray,
    ranks: numpy.ndarray,
    max_rank: int,
) -> "DataFrame":
    rows = []
    has_compound = "compound" in catalog.obs.columns
    for i, name in enumerate(classes.obs_names):
        for j in ranks[i].argsort():
            if ranks[i, j] > max_rank:
                break
            compound_classes = catalog.var.index[catalog.var_vector(catalog.obs.index[j]) > 0.5]
            bgc_classes = classes.var.index[classes.var_vector(name)]
            rows.append([
                name,
                ranks[i, j],
                catalog.obs.index[j],
                catalog.obs["compound"].iloc[j] if has_compound else "?",
                distances[i, j],
                ";".join(compound_classes),
                ";".join(bgc_classes),
            ])
    return pandas.DataFrame(
        rows,
        columns=[
            "bgc_id",
            "rank",
            "index",
            "compound",
            "distance",
            "compound_classes",
            "bgc_classes",
        ]
    )


def build_table(results: "DataFrame") -> rich.table.Table:
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


@requires("anndata")
def build_distances(
    X: numpy.ndarray, 
    obs: pandas.DataFrame, 
    var: pandas.DataFrame, 
    uns: Dict[str, object] = None
) -> "anndata.AnnData":
    if isinstance(X, numpy.matrix):
        X = numpy.asarray(X)
    return anndata.AnnData(X=X, obs=obs, var=var, uns=uns)


def probjaccard(x: numpy.ndarray, y: numpy.ndarray) -> float:
    tt = (x @ y).item()
    return 1.0 - tt / ( x.sum() + y.sum() - tt )


def probjaccard_cdist(X: numpy.ndarray, Y: numpy.ndarray) -> numpy.ndarray:
    tt = (X @ Y.T)
    return 1.0 - tt / (X.sum(axis=1).reshape(-1, 1) - tt + Y.sum(axis=1).reshape(1, -1))


@requires("scipy.stats")
def run(args: argparse.Namespace, console: Console) -> int:
    # load predictor
    predictor = load_model(args.model, console)
    probas, classes = load_predictions(args.input, predictor, console)
    uns = record_metadata(predictor)

    # load catalog
    catalog = load_catalog(args.catalog, predictor, console)[:, probas.var.index]

    # compute distance
    console.print(f"[bold blue]{'Computing':>12}[/] pairwise distances and ranks")
    distances = probjaccard_cdist(probas.X, catalog.X)
    distances = numpy.nan_to_num(distances, copy=False, nan=1.0)
    ranks = scipy.stats.rankdata(distances, method="dense", axis=1)

    # save results
    results = build_results(classes, catalog, distances, ranks, max_rank=args.rank)
    if args.output:
        console.print(f"[bold blue]{'Saving':>12}[/] search results to {str(args.output)!r}")
        results.to_csv(args.output, sep="\t", index=False)

    # save distance matrix
    if args.distance_matrix:
        dmatrix = build_distances(
            distances,
            obs=probas.obs,
            var=catalog.obs,
            uns=uns,
        )
        console.print(f"[bold blue]{'Saving':>12}[/] distance matrix {str(args.distance_matrix)!r}")
        dmatrix.write(args.distance_matrix)

    # display output
    if args.render:
        table = build_table(results)
        console.print(table)

    return 0
