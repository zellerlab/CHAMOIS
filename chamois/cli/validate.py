import argparse
import json
import math
import pathlib

import numpy
import rich.progress
from rich.console import Console

from .._meta import requires
from ..predictor import ChemicalOntologyPredictor
from ..predictor.information import information_accretion, information_theoric_curve, semantic_distance_score
from ..ontology import Ontology
from ._common import load_model, record_metadata, save_metrics, filter_dataset
from ._parser import (
    configure_group_preprocessing,
    configure_group_training_input,
    configure_group_hyperparameters,
    configure_group_cross_validation,
)


def configure_parser(parser: argparse.ArgumentParser):
    params_input = configure_group_training_input(parser)
    params_input.add_argument(
        "-m",
        "--model",
        default=None,
        type=pathlib.Path,
        help="The path to an alternative model to predict classes with."
    )

    configure_group_preprocessing(parser)
    params_output = parser.add_argument_group(
        'Output',
        'Mandatory and optional outputs.'
    )
    params_output.add_argument(
        "-o",
        "--output",
        required=True,
        type=pathlib.Path,
        help="The path where to write the computed probabilities."
    )
    params_output.add_argument(
        "--metrics",
        type=pathlib.Path,
        help="The path to an optional metrics file to write in DVC/JSON format."
    )
    params_output.add_argument(
        "--report",
        type=pathlib.Path,
        help="An optional file where to generate a label-wise evaluation report."
    )

    parser.set_defaults(run=run)


@requires("pandas")
@requires("sklearn.model_selection")
@requires("sklearn.feature_selection")
@requires("sklearn.metrics")
def run(args: argparse.Namespace, console: Console) -> int:
    import anndata

    # load data
    console.print(f"[bold blue]{'Loading':>12}[/] test data")
    features = anndata.concat([anndata.read_h5ad(file) for file in args.features], axis=1, merge="same")
    classes = anndata.read_h5ad(args.classes)
    console.print(f"[bold green]{'Loaded':>12}[/] {features.n_obs} observations, {features.n_vars} features and {classes.n_vars} classes")

    #
    features, classes = filter_dataset(
        features,
        classes,
        console,
        min_class_occurrences=0,
        min_feature_occurrences=0,
        min_genes=args.min_genes,
        min_length=args.min_cluster_length,
        fix_mismatch=args.mismatch,
    )

    # load model
    predictor = load_model(args.model, console=console)
    classes = classes[:, predictor.classes_.index]

    # convert features into model format
    # FIXME: use COO and only iterate on non-zero features
    console.print(f"[bold blue]{'Extracting':>12}[/] relevant features")
    coo = features.X.tocoo()
    X = numpy.zeros(( features.n_obs, len(predictor.features_) ))
    for (i, j, x) in zip(coo.row, coo.col, coo.data):
        try:
            k = predictor.features_.index.get_loc(features.var_names[j])
        except KeyError:
            pass
        else:
            X[i, k] = x

    # apply model
    probas = predictor.predict_probas(X)

    # compute AUROC for the entire classification
    ground_truth = classes.X.toarray()
    ia = information_accretion(ground_truth, predictor.ontology.adjacency_matrix)

    testable = numpy.zeros(ground_truth.shape[1], dtype=bool)
    for j in range(classes.n_vars):
        testable[j] = len(numpy.unique(ground_truth[:, j])) != 1

    ground_truth = ground_truth[:, testable]
    probas_prop = probas[:, testable]

    micro_auroc = sklearn.metrics.roc_auc_score(ground_truth, probas_prop, average="micro")
    macro_auroc = sklearn.metrics.roc_auc_score(ground_truth, probas_prop, average="macro")
    micro_avgpr = sklearn.metrics.average_precision_score(ground_truth, probas_prop, average="micro")
    macro_avgpr = sklearn.metrics.average_precision_score(ground_truth, probas_prop, average="macro")
    semdist = semantic_distance_score(ground_truth, probas_prop.round(3), ia[testable])
    jaccard = sklearn.metrics.jaccard_score(ground_truth, probas >= 0.5, average="samples")
    stats = [
        f"[bold magenta]AUROC(µ)=[/][bold cyan]{micro_auroc:05.1%}[/]",
        f"[bold magenta]AUROC(M)=[/][bold cyan]{macro_auroc:05.1%}[/]",
        f"[bold magenta]Avg.Precision(µ)=[/][bold cyan]{micro_avgpr:05.1%}[/]",
        f"[bold magenta]Avg.Precision(M)=[/][bold cyan]{macro_avgpr:05.1%}[/]",
        f"[bold magenta]SemanticDistance=[/][bold cyan]{semdist:.2f}[/]",
        f"[bold magenta]Jaccard=[/][bold cyan]{jaccard:04.3}[/]",
    ]
    console.print(f"[bold green]{'Finished':>12}[/] validation", *stats)

    # save metrics
    metrics = {
        "cv": {
            "AUROC(µ)": micro_auroc,
            "AUROC(M)": macro_auroc,
            "AveragePrecision(µ)": micro_avgpr,
            "AveragePrecision(M)": macro_avgpr,
            "SemanticDistance": semdist,
        },
        "variance": math.nan if predictor.variance is None else predictor.variance,
        "features": len(predictor.features_),
        "classes": len(predictor.classes_),
        "observations": features.n_obs,
    }
    save_metrics(metrics, args.metrics, console)

    # save predictions
    console.print(f"[bold blue]{'Saving':>12}[/] predictions to {str(args.output)!r}")
    if args.output.parent:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    data = anndata.AnnData(
        X=probas,
        obs=classes.obs,
        var=classes.var.assign(information_accretion=ia),
        uns=record_metadata()
    )
    data.write(args.output)
    console.print(f"[bold green]{'Finished':>12}[/] validating model")

    # generate report
    if args.report is not None:
        console.print(f"[bold blue]{'Generating':>12}[/] class-specific report")
        data = []
        preds = probas > 0.5
        for j in range(classes.n_vars):
            data.append({
                "class": classes.var_names[j],
                "average_precision": sklearn.metrics.average_precision_score(ground_truth[:, j], probas[:, j]),
                "auroc": sklearn.metrics.roc_auc_score(ground_truth[:, j], probas[:, j]),
                "f1_score": sklearn.metrics.f1_score(ground_truth[:, j], preds[:, j]),
                "hamming_loss": sklearn.metrics.hamming_loss(ground_truth[:, j], preds[:, j]),
                "accuracy_score": sklearn.metrics.hamming_loss(ground_truth[:, j], preds[:, j]),
                "precision": sklearn.metrics.precision_score(ground_truth[:, j], preds[:, j]),
                "recall": sklearn.metrics.recall_score(ground_truth[:, j], preds[:, j]),
                "balanced_accuracy": sklearn.metrics.balanced_accuracy_score(ground_truth[:, j], preds[:, j]),
                "adjusted_balanced_accuracy": sklearn.metrics.balanced_accuracy_score(ground_truth[:, j], preds[:, j], adjusted=True),
            })
        report = pandas.merge(classes.var, pandas.DataFrame(data), left_index=True, right_on="class")
        if args.report.parent:
            args.report.parent.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold blue]{'Saving':>12}[/] class-specific report to {str(args.report)!r}")
        report.to_csv(args.report, sep="\t", index=False)

    return 0