import argparse
import pathlib

import anndata
import numpy
import rich.progress
from rich.console import Console

from .._meta import requires
from ..ontology import Ontology
from ..predictor import ChemicalOntologyPredictor


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
    parser.add_argument(
        "--model",
        choices={"logistic", "ridge"},
        default="logistic",
        help="The kind of model to train."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="The strength of the parameters regularization.",
    )
    parser.set_defaults(run=run)

@requires("sklearn.metrics")
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
    ontology = Ontology(classes.varp["parents"].toarray())

    # traing model
    console.print(f"[bold blue]{'Training':>12}[/] logistic regression model")
    model = ChemicalOntologyPredictor(
        ontology,
        n_jobs=args.jobs,
        model=args.model,
        alpha=args.alpha,
    )
    model.fit(features, classes)

    # compute AUROC for classes that have positive and negative members
    # (scikit-learn will crash if a class only has positives/negatives)
    probas = model.predict_probas(features[:, model.features_.index])
    truth = classes.X.toarray()
    micro_auroc = sklearn.metrics.roc_auc_score(truth, probas, average="micro")
    macro_auroc = sklearn.metrics.roc_auc_score(truth, probas, average="macro")
    micro_avgpr = sklearn.metrics.average_precision_score(truth, probas, average="micro")
    macro_avgpr = sklearn.metrics.average_precision_score(truth, probas, average="macro")
    stats = [
        f"[bold magenta]AUROC(µ)=[/][bold cyan]{micro_auroc:05.1%}[/]",
        f"[bold magenta]AUROC(M)=[/][bold cyan]{macro_auroc:05.1%}[/]",
        f"[bold magenta]Avg.Precision(µ)=[/][bold cyan]{micro_avgpr:05.1%}[/]",
        f"[bold magenta]Avg.Precision(M)=[/][bold cyan]{macro_avgpr:05.1%}[/]",
    ]
    console.print(f"[bold green]{'Finished':>12}[/] training:", *stats)

    # save result
    console.print(f"[bold blue]{'Saving':>12}[/] trained model to {str(args.output)!r}")
    if args.output.parent:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as dst:
        model.save(dst)
    console.print(f"[bold green]{'Finished':>12}[/] training model")


