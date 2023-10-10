import argparse
import json
import math
import pathlib

import anndata
import numpy
import rich.progress
from rich.console import Console

from .._meta import requires
from ..predictor import ChemicalOntologyPredictor
from ..ontology import Ontology
from ._common import record_metadata, save_metrics


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
        help="The path where to write the probabilities for each test fold."
    )
    parser.add_argument(
        "--metrics",
        type=pathlib.Path,
        help="The path to an optional metrics file to write in DVC/JSON format."
    )
    parser.add_argument(
        "-k",
        "--kfolds",
        type=int,
        default=10,
        help="The number of cross-validation folds to run.",
    )
    parser.add_argument(
        "--sampling",
        choices={"random", "group", "kennard-stone"},
        default="group",
        help="The algorithm to use for partitioning folds.",
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
    parser.add_argument(
        "--variance",
        type=float,
        help="The variance threshold for filtering features.",
    )
    parser.set_defaults(run=run)


@requires("sklearn.model_selection")
@requires("sklearn.feature_selection")
@requires("sklearn.metrics")
@requires("kennard_stone")
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
    classes = classes[:, (classes.X.sum(axis=0).A1 >= 5) & (classes.X.sum(axis=0).A1 <= classes.n_obs - 5)]
    console.print(f"[bold blue]{'Using':>12}[/] {classes.n_vars} nontautological classes")
    # prepare ontology and groups
    ontology = Ontology(classes.varp["parents"])
    groups = classes.obs["compound"].cat.codes

    # start training
    console.print(f"[bold blue]{'Splitting':>12}[/] data into {args.kfolds} folds")
    if args.sampling == "group":
        kfold = sklearn.model_selection.GroupShuffleSplit(n_splits=args.kfolds, random_state=args.seed)
    elif args.sampling == "random":
        kfold = sklearn.model_selection.KFold(n_splits=args.kfolds, random_state=args.seed, shuffle=True)
    elif args.sampling == "kennard-stone":
        kfold = kennard_stone.KFold(n_splits=args.kfolds, n_jobs=args.jobs, metric="cosine")
    else:
        raise ValueError(f"Invalid value for `--sampling`: {args.sampling!r}")
    splits = list(kfold.split(features.X.toarray(), classes.X.toarray(), groups))

    console.print(f"[bold blue]{'Running':>12}[/] cross-validation evaluation")
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
        )
        model.fit(train_X, train_Y)

        # test fold
        test_X = features[test_indices, model.features_.index]
        test_Y = classes[test_indices, model.classes_.index].X.toarray()
        probas[test_indices] = model.predict_probas(test_X)
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

    # compute AUROC for the entire classification
    micro_auroc = sklearn.metrics.roc_auc_score(classes.X.toarray(), probas, average="micro")
    macro_auroc = sklearn.metrics.roc_auc_score(classes.X.toarray(), probas, average="macro")
    micro_avgpr = sklearn.metrics.average_precision_score(classes.X.toarray(), probas, average="micro")
    macro_avgpr = sklearn.metrics.average_precision_score(classes.X.toarray(), probas, average="macro")
    stats = [
        f"[bold magenta]AUROC(µ)=[/][bold cyan]{micro_auroc:05.1%}[/]",
        f"[bold magenta]AUROC(M)=[/][bold cyan]{macro_auroc:05.1%}[/]",
        f"[bold magenta]Avg.Precision(µ)=[/][bold cyan]{micro_avgpr:05.1%}[/]",
        f"[bold magenta]Avg.Precision(M)=[/][bold cyan]{macro_avgpr:05.1%}[/]",
    ]
    console.print(f"[bold green]{'Finished':>12}[/] cross-validation", *stats)

    # save metrics
    metrics = {
        "cv": {
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

    # save predictions
    console.print(f"[bold blue]{'Saving':>12}[/] predictions to {str(args.output)!r}")
    if args.output.parent:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    data = anndata.AnnData(
        X=probas,
        obs=classes.obs,
        var=classes.var,
        uns=record_metadata(),
    )
    data.write(args.output)
    console.print(f"[bold green]{'Finished':>12}[/] cross-validating model")

   
