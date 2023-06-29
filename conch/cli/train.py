import argparse
import pathlib

import anndata
import rich.progress
from rich.console import Console

from .._meta import requires
from ..treematrix import TreeMatrix
from ..predictor import ChemicalHierarchyPredictor


def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-f",
        "--features",
        required=True,
        type=pathlib.Path,
        help="The feature table in HDF5 format to use for training the predictor."
    )
    parser.add_argument(
        "-c",
        "--classes",
        required=True,
        type=pathlib.Path,
        help="The classes table in HDF5 format to use for training the predictor."
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=pathlib.Path,
        help="The path where to write the trained model in pickle format."
    )
    parser.add_argument(
        "--min-occurences",
        type=int,
        default=3,
        help="The minimum of occurences for a feature to be retained."
    )
    parser.set_defaults(run=run)

@requires("sklearn")
def run(args: argparse.Namespace, console: Console) -> int:
    # load data
    console.print(f"[bold blue]{'Loading':>12}[/] training data")
    features = anndata.read(args.features)
    classes = anndata.read(args.classes)
    # remove compounds with unknown structure
    features = features[~classes.obs.unknown_structure]
    classes = classes[~classes.obs.unknown_structure]
    # remove features absent from training set
    features = features[:, features.X.sum(axis=0).A1 >= args.min_occurences]
    # remove clases absent from training set
    classes = classes[:, (classes.X.sum(axis=0).A1 >= 5) & (classes.X.sum(axis=0).A1 <= classes.n_obs - 5)]
    # prepare class hierarchy
    hierarchy = classes.varp["parents"].toarray()

    console.print(f"[bold blue]{'Training':>12}[/] logistic regression model")
    model = ChemicalHierarchyPredictor(hierarchy=TreeMatrix(hierarchy), n_jobs=args.jobs)
    model.fit(features, classes)

    # save result
    console.print(f"[bold blue]{'Saving':>12}[/] trained model to {str(args.output)!r}")
    if args.output.parent:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as dst:
        model.save(dst)
    console.print(f"[bold green]{'Finished':>12}[/] training model")


