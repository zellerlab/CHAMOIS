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
from ._common import save_metrics, filter_dataset
from ._parser import (
    configure_group_preprocessing,
    configure_group_training_input,
    configure_group_hyperparameters,
)


def configure_parser(parser: argparse.ArgumentParser):
    configure_group_training_input(parser)
    configure_group_preprocessing(parser)
    configure_group_hyperparameters(parser)

    params_output = parser.add_argument_group(
        'Output',
        'Mandatory and optional outputs.'
    )
    params_output.add_argument(
        "-o",
        "--output",
        required=True,
        type=pathlib.Path,
        help="The path where to write the trained model in JSON format."
    )
    params_output.add_argument(
        "--metrics",
        type=pathlib.Path,
        help="The path to an optional metrics file to write in DVC/JSON format."
    )

    parser.set_defaults(run=run)


@requires("sklearn.metrics")
def run(args: argparse.Namespace, console: Console) -> int:
    # load data
    console.print(f"[bold blue]{'Loading':>12}[/] training data")
    features = anndata.concat([anndata.read(file) for file in args.features], axis=1, merge="same")
    classes = anndata.read(args.classes)
    console.print(f"[bold green]{'Loaded':>12}[/] {features.n_obs} observations, {features.n_vars} features and {classes.n_vars} classes")

    # preprocess data
    similarity = None if args.similarity is None else anndata.read(args.similarity)
    features, classes = filter_dataset(
        features,
        classes,
        console,
        similarity=similarity,
        remove_unknown_structure=True,
        min_class_occurrences=args.min_class_occurrences,
        min_feature_occurrences=args.min_feature_occurrences,
        min_length=args.min_cluster_length,
        min_genes=args.min_genes,
    )
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
    macro_avgpr = sklearn.metrics.average_precision_score(truth, probas.round(3), average="macro")
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
