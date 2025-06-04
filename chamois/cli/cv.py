import argparse
import json
import math
import pathlib
import typing

import numpy
import rich.progress
from rich.console import Console

from .._meta import requires
from ..predictor import ChemicalOntologyPredictor
from ..predictor.information import information_accretion, information_theoric_curve, semantic_distance_score
from ..ontology import Ontology
from ._common import record_metadata, save_metrics, filter_dataset
from ._parser import (
    configure_group_preprocessing,
    configure_group_training_input,
    configure_group_hyperparameters,
    configure_group_cross_validation,
)

if typing.TYPE_CHECKING:
    import kennard_stone


def configure_parser(parser: argparse.ArgumentParser):
    configure_group_training_input(parser)
    configure_group_preprocessing(parser)
    configure_group_hyperparameters(parser)
    configure_group_cross_validation(parser)

    params_output = parser.add_argument_group(
        'Output', 
        'Mandatory and optional outputs.'
    )
    params_output.add_argument(
        "-o",
        "--output",
        required=True,
        type=pathlib.Path,
        help="The path where to write the probabilities for each test fold."
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
    params_output.add_argument(
        "--best-model",
        type=pathlib.Path,
        help="An optional file where to write the model with highest macro-average-precision."
    )

    parser.set_defaults(run=run)


@requires("kennard_stone")
def kennard_stone_kfold(n_splits: int, n_jobs: int, metric: str) -> "kennard_stone.KFold":
    return kennard_stone.KFold(n_splits=n_splits, n_jobs=n_job, metric=metric)


@requires("anndata")
@requires("pandas")
@requires("sklearn.model_selection")
@requires("sklearn.feature_selection")
@requires("sklearn.metrics")
def run(args: argparse.Namespace, console: Console) -> int:
    # load data
    console.print(f"[bold blue]{'Loading':>12}[/] training data")
    features = anndata.concat([anndata.read_h5ad(file) for file in args.features], axis=1, merge="first")
    classes = anndata.read_h5ad(args.classes)
    console.print(f"[bold green]{'Loaded':>12}[/] {features.n_obs} observations, {features.n_vars} features and {classes.n_vars} classes")
    
    # preprocess data
    similarity = None if args.similarity is None else anndata.read_h5ad(args.similarity)
    features, classes = filter_dataset(
        features,
        classes,
        console,
        similarity=similarity,
        remove_unknown_structure=True,
        min_class_occurrences=args.min_class_occurrences,
        min_feature_occurrences=args.min_feature_occurrences,
        min_class_groups=args.min_class_groups,
        min_feature_groups=args.min_feature_groups,
        min_length=args.min_cluster_length,
        min_genes=args.min_genes,
        fix_mismatch=args.mismatch,
    )
    
    # prepare ontology and groups
    ontology = Ontology(classes.varp["parents"])
    groups = None

    # start training
    ground_truth = classes.X.toarray()
    console.print(f"[bold blue]{'Splitting':>12}[/] data into {args.kfolds} folds")
    if args.sampling == "group":
        groups = classes.obs["compound"].cat.codes
        kfold = sklearn.model_selection.GroupShuffleSplit(n_splits=args.kfolds, random_state=args.seed)
    elif args.sampling == "random":
        kfold = sklearn.model_selection.KFold(n_splits=args.kfolds, random_state=args.seed, shuffle=True)
    elif args.sampling == "kennard-stone":
        kfold = kennard_stone_kfold(n_splits=args.kfolds, n_jobs=args.jobs, metric="cosine")
    else:
        raise ValueError(f"Invalid value for `--sampling`: {args.sampling!r}")
    splits = list(kfold.split(features.X.toarray(), ground_truth, groups))

    console.print(f"[bold blue]{'Running':>12}[/] cross-validation evaluation")
    best_model = None
    best_avgpr = 0.0
    probas = numpy.zeros(classes.X.shape, dtype=float)
    for i, (train_indices, test_indices) in enumerate(splits):
        # extract fold observations
        train_X = features[train_indices]
        train_Y = classes[train_indices]
        # train fold
        model = ChemicalOntologyPredictor(
            ontology,
            n_jobs=args.jobs,
            model=args.model,
            alpha=args.alpha,
            variance=args.variance,
            seed=args.seed,
        )
        model.fit(train_X, train_Y)
        # test fold
        test_X = features[test_indices, model.features_.index]
        test_Y = classes[test_indices, model.classes_.index].X.toarray()
        probas[test_indices] = model.predict_probas(test_X, propagate=False)
        # compute AUROC for classes that have positive and negative members
        # (scikit-learn will crash if a class only has positives/negatives)
        mask = ~numpy.all(test_Y == test_Y[0], axis=0)
        micro_auroc = sklearn.metrics.roc_auc_score(test_Y[:, mask], probas[test_indices][:, mask], average="micro")
        macro_auroc = sklearn.metrics.roc_auc_score(test_Y[:, mask], probas[test_indices][:, mask], average="macro")
        micro_avgpr = sklearn.metrics.average_precision_score(test_Y[:, mask], probas[test_indices][:, mask], average="micro")
        macro_avgpr = sklearn.metrics.average_precision_score(test_Y[:, mask], probas[test_indices][:, mask], average="macro")
        stats = [
            f"[bold magenta]AUROC(µ)=[/][bold cyan]{micro_auroc:05.1%}[/]",
            f"[bold magenta]AUROC(M)=[/][bold cyan]{macro_auroc:05.1%}[/]",
            f"[bold magenta]Avg.Precision(µ)=[/][bold cyan]{micro_avgpr:05.1%}[/]",
            f"[bold magenta]Avg.Precision(M)=[/][bold cyan]{macro_avgpr:05.1%}[/]",
        ]
        console.print(f"[bold green]{'Finished':>12}[/] fold {i+1:2}:", *stats)
        if best_model is None or macro_avgpr > best_avgpr:
            best_model = model
            best_avgpr = macro_avgpr

    # compute AUROC for the entire classification
    ia = information_accretion(ground_truth, ontology.adjacency_matrix)
    probas_prop = model.propagate(probas)
    micro_auroc = sklearn.metrics.roc_auc_score(ground_truth, probas_prop, average="micro")
    macro_auroc = sklearn.metrics.roc_auc_score(ground_truth, probas_prop, average="macro")
    micro_avgpr = sklearn.metrics.average_precision_score(ground_truth, probas_prop, average="micro")
    macro_avgpr = sklearn.metrics.average_precision_score(ground_truth, probas_prop, average="macro")
    semdist = semantic_distance_score(ground_truth, probas_prop.round(3), ia)    
    jaccard = sklearn.metrics.jaccard_score(ground_truth, probas >= 0.5, average="samples")
    stats = [
        f"[bold magenta]AUROC(µ)=[/][bold cyan]{micro_auroc:05.1%}[/]",
        f"[bold magenta]AUROC(M)=[/][bold cyan]{macro_auroc:05.1%}[/]",
        f"[bold magenta]Avg.Precision(µ)=[/][bold cyan]{micro_avgpr:05.1%}[/]",
        f"[bold magenta]Avg.Precision(M)=[/][bold cyan]{macro_avgpr:05.1%}[/]",
        f"[bold magenta]SemanticDistance=[/][bold cyan]{semdist:.2f}[/]",
        f"[bold magenta]Jaccard=[/][bold cyan]{jaccard:04.3}[/]",
    ]
    console.print(f"[bold green]{'Finished':>12}[/] cross-validation", *stats)

    # save metrics
    metrics = {
        "cv": {
            "AUROC(µ)": micro_auroc,
            "AUROC(M)": macro_auroc,
            "AveragePrecision(µ)": micro_avgpr,
            "AveragePrecision(M)": macro_avgpr,
            "SemanticDistance": semdist,
        },
        "variance": math.nan if model.variance is None else model.variance,
        "features": len(model.features_),
        "classes": len(model.classes_),
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
    console.print(f"[bold green]{'Finished':>12}[/] cross-validating model")

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

    # save best model
    if args.best_model:
        console.print(f"[bold blue]{'Saving':>12}[/] best model to {str(args.best_model)!r}")
        if args.best_model.parent:
            args.best_model.parent.mkdir(parents=True, exist_ok=True)
        with args.best_model.open("w") as dst:
            best_model.save(dst)

    return 0