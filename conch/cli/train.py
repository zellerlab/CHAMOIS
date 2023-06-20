import argparse
import pathlib

import anndata
import rich.progress
from rich.console import Console

from ..treematrix import TreeMatrix
from ..predictor import ChemicalHierarchyPredictor


def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument("-f", "--features", required=True, type=pathlib.Path)
    parser.add_argument("-c", "--classes", required=True, type=pathlib.Path)
    parser.add_argument("-o", "--output", required=True, type=pathlib.Path)
    parser.add_argument("-j", "--jobs", type=int, default=None)
    # parser.add_argument("-e", "--epochs", type=int, default=200, help="The number of epochs to train the model for.")
    # parser.add_argument("--report-period", type=int, default=20, help="Report evaluation metrics every N iterations.")
    parser.set_defaults(run=run)


def run(args: argparse.Namespace, console: Console) -> int:
    # load data
    console.print(f"[bold blue]{'Loading':>12}[/] training data")
    features = anndata.read(args.features)
    classes = anndata.read(args.classes)
    # remove compounds with unknown structure
    features = features[~classes.obs.unknown_structure]
    classes = classes[~classes.obs.unknown_structure]
    # remove features absent from training set
    features = features[:, features.X.sum(axis=0).A1 > 0]
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
    with open(args.output, "wb") as dst:
        model.save(dst)
    console.print(f"[bold green]{'Finished':>12}[/] training model")


