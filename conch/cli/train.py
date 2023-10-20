import argparse
import json
import math
import pathlib

import anndata
import numpy
import rich.progress
from rich.console import Console

from .._meta import requires
from ..ontology import Ontology
from ..predictor import ChemicalOntologyPredictor
from ._common import save_metrics


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
        "-s",
        "--similarity",
        type=pathlib.Path,
        help="Pairwise nucleotide similarities for deduplication the observations."
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=pathlib.Path,
        help="The path where to write the trained model in pickle format."
    )
    parser.add_argument(
        "--metrics",
        type=pathlib.Path,
        help="The path to an optional metrics file to write in DVC/JSON format."
    )
    parser.add_argument(
        "--min-class-occurrences",
        type=int,
        default=10,
        help="The minimum of occurences for a class to be retained."
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
    parser.add_argument(
        "--variance",
        type=float,
        help="The variance threshold for filtering features.",
        default=None,
    )
    parser.set_defaults(run=run)

@requires("sklearn.metrics")
def run(args: argparse.Namespace, console: Console) -> int:
    # load data
    console.print(f"[bold blue]{'Loading':>12}[/] training data")
    features = anndata.read(args.features)
    classes = anndata.read(args.classes)
    console.print(f"[bold green]{'Loaded':>12}[/] {features.n_obs} observations, {features.n_vars} features and {classes.n_vars} classes")
    # remove compounds with unknown structure
    features = features[~classes.obs.unknown_structure]
    classes = classes[~classes.obs.unknown_structure]
    console.print(f"[bold blue]{'Using':>12}[/] {features.n_obs} observations with known compounds")
    # remove similar BGCs based on nucleotide similarity
    if args.similarity is not None:
        ani = anndata.read(args.similarity).obs
        ani = ani.loc[classes.obs_names].drop_duplicates("groups")
        classes = classes[ani.index]
        features = features[ani.index]
        console.print(f"[bold blue]{'Using':>12}[/] {features.n_obs} unique observations based on nucleotide similarity")
    # remove classes absent from training set
    support = classes.X.sum(axis=0).A1
    classes = classes[:, (support >= args.min_class_occurrences) & (support <= classes.n_obs - args.min_class_occurrences)]
    console.print(f"[bold blue]{'Using':>12}[/] {classes.n_vars} classes with at least {args.min_class_occurrences} members")
    # prepare class hierarchy
    ontology = Ontology(classes.varp["parents"].toarray())

    # traing model
    console.print(f"[bold blue]{'Training':>12}[/] logistic regression model")
    model = ChemicalOntologyPredictor(
        ontology,
        n_jobs=args.jobs,
        model=args.model,
        alpha=args.alpha,
        variance=args.variance,
    )
    model.fit(features, classes)
    console.print(f"[bold blue]{'Retaining':>12}[/] {len(model.features_)} features in final model")

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

    # save metrics
    metrics = {
        "training": {
            "AUROC(µ)": micro_auroc,
            "AUROC(M)": macro_auroc,
            "AveragePrecision(µ)": micro_avgpr,
            "AveragePrecision(M)": macro_avgpr,
        },
        "variance": math.nan if model.variance is None else model.variance,
        "features": len(model.features_),
        "classes": len(model.classes_),
        "observations": features.n_obs,
    }
    save_metrics(metrics, args.metrics, console)
    
    # save result
    console.print(f"[bold blue]{'Saving':>12}[/] trained model to {str(args.output)!r}")
    if args.output.parent:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as dst:
        model.save(dst)
    console.print(f"[bold green]{'Finished':>12}[/] training model")
