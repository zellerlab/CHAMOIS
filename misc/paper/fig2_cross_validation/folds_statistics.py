import argparse
import collections
import itertools
import json
import math
import os
import pathlib
import sys
import random

import anndata
import numpy
import pandas
import joblib
import rich.progress
import rdkit.Chem
import sklearn.model_selection
import sklearn.dummy
import scipy.spatial.distance
from rdkit import RDLogger
from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder
from sklearn.metrics import precision_recall_curve
from rich.console import Console
from matplotlib import rcParams
from matplotlib import pyplot as plt

folder = pathlib.Path(__file__).parent
PROJECT_FOLDER = folder
while not PROJECT_FOLDER.joinpath("chamois").exists():
    PROJECT_FOLDER = PROJECT_FOLDER.parent
sys.path.insert(0, str(PROJECT_FOLDER))

from chamois._meta import requires
from chamois.predictor import ChemicalOntologyPredictor
from chamois.predictor.information import information_accretion, information_theoric_curve, semantic_distance_score

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--features", required=True)
parser.add_argument("-c", "--classes", required=True)
parser.add_argument("--report", required=True)
# parser.add_argument("-s", "--similarity", required=True)
# parser.add_argument("--report", type=pathlib.Path)
parser.add_argument("-o", "--output", required=True, type=pathlib.Path)
parser.add_argument("-k", "--kfolds", type=int, default=5)
parser.add_argument("-j", "--jobs", type=int, default=-1)
parser.add_argument("--taxonomy", type=pathlib.Path, default=None)
# parser.add_argument("--model", choices=ChemicalOntologyPredictor._MODELS, default="logistic")
parser.add_argument("--seed", default=0, type=int)
args = parser.parse_args()

rcParams['svg.fonttype'] = 'none'
parallel = joblib.Parallel(n_jobs=args.jobs)
random.seed(args.seed)

# disable logging
RDLogger.DisableLog('rdApp.warning')

console = Console()
console.print(f"[bold blue]{'Loading':>12}[/] training data")
features = anndata.read_h5ad(args.features)
classes = anndata.read_h5ad(args.classes)
console.print(f"[bold green]{'Loaded':>12}[/] {features.n_obs} observations, {features.n_vars} features and {classes.n_vars} classes")

# remove compounds not in Bacteria
if args.taxonomy:
    taxonomy = pandas.read_table(args.taxonomy)
    obs = pandas.merge(classes.obs, taxonomy, left_index=True, right_on="bgc_id")
    features = features[obs[obs.superkingdom == "Bacteria"].bgc_id]
    classes = classes[obs[obs.superkingdom == "Bacteria"].bgc_id]
    console.print(f"[bold green]{'Loaded':>12}[/] {features.n_obs} observations, {features.n_vars} features and {classes.n_vars} classes")

# load cross-validation report
console.print(f"[bold blue]{'Loading':>12}[/] cross-validation report")
cv = pandas.read_table(args.report).set_index('class')

# remove compounds with unknown structure
features = features[~classes.obs.unknown_structure]
classes = classes[~classes.obs.unknown_structure]
console.print(f"[bold blue]{'Using':>12}[/] {features.n_obs} observations with known compounds")

# remove classes absent from the cross-validation
classes = classes[:, cv.index]
console.print(f"[bold blue]{'Using':>12}[/] {classes.n_vars} classes used in cross-validation")

# prepare ontology and groups
groups = classes.obs["groups"]

# compute MHFP6 fingerprints
encoder = MHFPEncoder(2048, 42)
console.print(f"[bold blue]{'Computing':>12}[/] MHFP6 fingerprints for {classes.n_obs} compounds")
mhfp6 = numpy.asarray(encoder.EncodeSmilesBulk(list(classes.obs.smiles), kekulize=True))

# compute pairwise distances
cdist = 1.0 - scipy.spatial.distance.cdist(mhfp6, mhfp6, metric="hamming")
# get ANI
# ani = anndata.read_h5ad(args.similarity)
# ani = ani[classes.obs_names].X.toarray()

# extract feature kinds
kinds = sorted(features.var.kind.unique())
console.print(f"[bold green]{'Found':>12}[/] unique feature kinds: [bold cyan]{'[/], [bold cyan]'.join(kinds)}[/]")

# prepare folds
ground_truth = classes.X.toarray()
console.print(f"[bold blue]{'Splitting':>12}[/] data into {args.kfolds} folds")
sgkfold = sklearn.model_selection.StratifiedGroupKFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)
kfold = sklearn.model_selection.KFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)


# 
plt.figure(figsize=(3, 3))

# compute empirical sequence similarity
dist_kfold = []
dist_sgkfold = []
X = numpy.zeros((features.n_obs, 1))
for class_index in rich.progress.track(range(classes.n_vars), console=console, description=f"[bold blue]{'Working':>12}[/]"):
    # console.print(f"[bold blue]{'Evaluating':>12}[/] class [bold cyan]{classes.var_names[class_index]}[/] ({classes.var.name[class_index]!r})")
    splits_kfold = list(kfold.split(X, ground_truth[:, class_index]))
    splits_sgkfold = list(sgkfold.split(X, ground_truth[:, class_index], groups))

    for i, (train_indices, test_indices) in enumerate(splits_kfold):
        train_mhfp6 = cdist[test_indices, :][:, train_indices].max(axis=0)
        dist_kfold.extend(train_mhfp6)
        
    #     # train_mask = numpy.zeros(classes.n_obs, dtype=bool)
    #     # train_mask[train_indices] = True
    #     # test_mask = numpy.zeros(classes.n_obs, dtype=bool)
    #     # test_mask[test_indices] = True

    #     # test_mhfp6 = cdist[test_indices, :][:, test_indices].copy()
    #     # test_mhfp6[test_mhfp6 == 1.0] = 0.0

    #     train_mhfp6 = cdist[test_indices, :][:, train_indices].copy()
    #     # train_mhfp6[train_mhfp6 == 1.0] = 0.0

    #     dist_kfold.extend(train_mhfp6.max(axis=0))
    #     # numpy.fill_diagonal(test_mhfp6, 0.0)

    #     # inner.extend(test_mhfp6.max(axis=0))
    #     # outer.extend(train_mhfp6.max(axis=0))
        
    #     # test_mhfp6 = mhfp6[test_indices]
    #     # train_mhfp6 = mhfp6[train_indices]

    #     # # for compount in classes.obs.iloc[train_indices]:

    #     # # inner.extend(scipy.spatial.distance.cdist(test_mhfp6, test_mhfp6, metric="hamming").min(axis=0))
    #     # outer.extend(.min(axis=0))

    #     # # print(inner.mean(), outer.mean())

    for i, (train_indices, test_indices) in enumerate(splits_sgkfold):
        train_mhfp6 = cdist[test_indices, :][:, train_indices].max(axis=0)
        dist_sgkfold.extend(train_mhfp6)
        # train_ani = ani[test_indices, :][:, train_indices].max(axis=0)
        # plt.scatter(train_ani, train_mhfp6)

# plt.show()
bins = numpy.linspace(0.0, 1, 20)
plt.hist(dist_kfold, bins=bins, label="K-fold", alpha=0.5, density=True, stacked=True)
plt.hist(dist_sgkfold, bins=bins, label="Stratified Group K-fold", alpha=0.5, density=True, stacked=True)
plt.xlabel("Maximum Chemical similarity")
plt.legend()
plt.savefig(args.output)
plt.show()
