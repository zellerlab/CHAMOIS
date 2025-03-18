import argparse
import collections
import itertools
import json
import math
import os
import pathlib
import sys

import anndata
import numpy
import pandas
import joblib
import rich.progress
import sklearn.model_selection
import sklearn.dummy
from sklearn.metrics import precision_recall_curve
from rich.console import Console

sys.path.insert(0, str(pathlib.Path(__file__).parents[4])) #os.path.realpath(os.path.join(__file__, "..", "..", "..")))

from chamois._meta import requires
from chamois.predictor import ChemicalOntologyPredictor
from chamois.predictor.information import information_accretion, information_theoric_curve, semantic_distance_score
from chamois.ontology import Ontology

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--features", required=True)
parser.add_argument("-c", "--classes", required=True)
parser.add_argument("--report", type=pathlib.Path)
parser.add_argument("-o", "--output", required=True, type=pathlib.Path)
parser.add_argument("-k", "--kfolds", type=int, default=5)
parser.add_argument("-j", "--jobs", type=int, default=-1)
parser.add_argument("--taxonomy", type=pathlib.Path, default=None)
parser.add_argument("--model", choices=ChemicalOntologyPredictor._MODELS, default="logistic")
parser.add_argument("--min-class-occurrences", type=int, default=10)
parser.add_argument("--seed", default=0, type=int)
args = parser.parse_args()

parallel = joblib.Parallel(n_jobs=args.jobs)

console = Console()
console.print(f"[bold blue]{'Loading':>12}[/] training data")
features = anndata.read(args.features)
classes = anndata.read(args.classes)
console.print(f"[bold green]{'Loaded':>12}[/] {features.n_obs} observations, {features.n_vars} features and {classes.n_vars} classes")

# remove compounds not in Bacteria
if args.taxonomy:
    taxonomy = pandas.read_table(args.taxonomy)
    obs = pandas.merge(classes.obs, taxonomy, left_index=True, right_on="bgc_id")
    features = features[obs[obs.superkingdom == "Bacteria"].bgc_id]
    classes = classes[obs[obs.superkingdom == "Bacteria"].bgc_id]
    console.print(f"[bold green]{'Loaded':>12}[/] {features.n_obs} observations, {features.n_vars} features and {classes.n_vars} classes")

# remove compounds with unknown structure
features = features[~classes.obs.unknown_structure]
classes = classes[~classes.obs.unknown_structure]
console.print(f"[bold blue]{'Using':>12}[/] {features.n_obs} observations with known compounds")
# remove classes absent from training set
support = classes.X.sum(axis=0).A1
classes = classes[:, (support >= args.min_class_occurrences) & (support <= classes.n_obs - args.min_class_occurrences)]
console.print(f"[bold blue]{'Using':>12}[/] {classes.n_vars} classes with at least {args.min_class_occurrences} members")
# prepare ontology and groups
ontology = Ontology(classes.varp["parents"])
# groups = classes.obs["compound"].cat.codes
groups = classes.obs["groups"]

# extract feature kinds
kinds = sorted(features.var.kind.unique())
console.print(f"[bold green]{'Found':>12}[/] unique feature kinds: [bold cyan]{'[/], [bold cyan]'.join(kinds)}[/]")

# start training
ground_truth = classes.X.toarray()
console.print(f"[bold blue]{'Splitting':>12}[/] data into {args.kfolds} folds")
kfold = sklearn.model_selection.StratifiedGroupKFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)
# splits = list(kfold.split(features.X.toarray(), ground_truth, groups))

@joblib.delayed
def runcv(i, class_index, train_indices, test_indices):
    train_X = features.X[train_indices].toarray()
    train_Y = classes.X[train_indices].toarray()[:, class_index]
    test_X = features.X[test_indices].toarray()
    try:
        if args.model == "logistic":
            model = sklearn.linear_model.LogisticRegression(
                "l1",
                solver="liblinear",
                max_iter=100,
                C=1.0,
            )
        elif args.model == "ridge":
            model = sklearn.linear_model.LogisticRegression(
                "l2",
                solver="liblinear",
                max_iter=100,
                C=1.0,
            )
        elif args.model == "dummy":
            model = sklearn.dummy.DummyClassifier()
        model.fit(train_X, train_Y)
        probas = model.predict_proba(test_X)
        return probas[:, 1]
    except (ValueError, IndexError):
        return numpy.array(train_Y[0]).repeat(test_indices.shape[0])

probas = numpy.zeros(classes.X.shape, dtype=float)
for class_index in rich.progress.track(range(classes.n_vars), console=console, description=f"[bold blue]{'Working':>12}[/]"):
    console.print(f"[bold blue]{'Evaluating':>12}[/] class [bold cyan]{classes.var_names[class_index]}[/] ({classes.var.name.iloc[class_index]!r})")
    splits = list(kfold.split(features.X.toarray(), ground_truth[:, class_index], groups))
    results = parallel(
        runcv(
            i, 
            class_index, 
            train_indices, 
            test_indices
        ) 
        for i, (train_indices, test_indices) in enumerate(splits)
    )
    for (_, test_indices), result in zip(splits, results):
        probas[test_indices, class_index] = result

# compute AUROC for the entire classification
ia = information_accretion(ground_truth, ontology.adjacency_matrix)
micro_auroc = sklearn.metrics.roc_auc_score(ground_truth, probas, average="micro")
macro_auroc = sklearn.metrics.roc_auc_score(ground_truth, probas, average="macro")
micro_avgpr = sklearn.metrics.average_precision_score(ground_truth, probas, average="micro")
macro_avgpr = sklearn.metrics.average_precision_score(ground_truth, probas, average="macro")
semdist = semantic_distance_score(ground_truth, probas.round(3), ia)    
stats = [
    f"[bold magenta]AUROC(µ)=[/][bold cyan]{micro_auroc:05.1%}[/]",
    f"[bold magenta]AUROC(M)=[/][bold cyan]{macro_auroc:05.1%}[/]",
    f"[bold magenta]Avg.Precision(µ)=[/][bold cyan]{micro_avgpr:05.1%}[/]",
    f"[bold magenta]Avg.Precision(M)=[/][bold cyan]{macro_avgpr:05.1%}[/]",
    f"[bold magenta]SemanticDistance=[/][bold cyan]{semdist:.2f}[/]",
]
console.print(f"[bold green]{'Finished':>12}[/] cross-validation", *stats)

# save predictions
console.print(f"[bold blue]{'Saving':>12}[/] predictions to {str(args.output)!r}")
if args.output.parent:
    args.output.parent.mkdir(parents=True, exist_ok=True)
data = anndata.AnnData(
    X=probas, 
    obs=classes.obs, 
    var=classes.var.assign(information_accretion=ia), 
)
data.write(args.output)
console.print(f"[bold green]{'Finished':>12}[/] cross-validating model")

# generate report
if args.report is not None:
    console.print(f"[bold blue]{'Generating':>12}[/] class-specific report")
    data = []
    preds = probas > 0.5
    for j in range(classes.n_vars):
        precision, recall, thresholds = precision_recall_curve(ground_truth[:, j], probas[:, j])
        f1score = (2 * precision * recall) / (precision + recall + 1e-10) 
        optimal = f1score.argmax()
        default = numpy.abs(thresholds - 0.5).argmin()
        data.append({
            "class": classes.var_names[j],
            "auprc": sklearn.metrics.average_precision_score(ground_truth[:, j], probas[:, j]),
            "auroc": sklearn.metrics.roc_auc_score(ground_truth[:, j], probas[:, j]),
            "f1_score": sklearn.metrics.f1_score(ground_truth[:, j], preds[:, j]),
            "hamming_loss": sklearn.metrics.hamming_loss(ground_truth[:, j], preds[:, j]),
            "accuracy_score": sklearn.metrics.hamming_loss(ground_truth[:, j], preds[:, j]),
            "precision": sklearn.metrics.precision_score(ground_truth[:, j], preds[:, j]),
            "recall": sklearn.metrics.recall_score(ground_truth[:, j], preds[:, j]),
            "balanced_accuracy": sklearn.metrics.balanced_accuracy_score(ground_truth[:, j], preds[:, j]),
            "adjusted_balanced_accuracy": sklearn.metrics.balanced_accuracy_score(ground_truth[:, j], preds[:, j], adjusted=True),
            "optimal_threshold_f1_score": f1score[optimal],
            "optimal_threshold_probability": thresholds[optimal],
            "default_threshold_f1_score": f1score[default],
        })
    report = pandas.merge(classes.var, pandas.DataFrame(data), left_index=True, right_on="class")
    if args.report.parent:
        args.report.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold blue]{'Saving':>12}[/] class-specific report to {str(args.report)!r}")
    report.to_csv(args.report, sep="\t", index=False)
