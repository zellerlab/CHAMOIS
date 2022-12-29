import argparse
import itertools
import json
import gzip
import csv
import pickle
import os

import joblib
import gb_io
import anndata
import disjoint_set
import fisher
import numpy
import pandas
import pronto
import scipy.sparse
import rich.progress
import rich.table
import rich.tree
import rich.panel
import sklearn.tree
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.multiclass
import sklearn.neural_network
import sklearn.pipeline
import sklearn.linear_model
import sklearn.neighbors
import pyrodigal
import pyhmmer
from palettable.cartocolors.qualitative import Bold_10

parallel = joblib.Parallel(backend="threading", n_jobs=-1)



# --- Load ChemOnt ------------------------------------------------------------

chemont = pronto.Ontology("data/chemont/ChemOnt_2_1.obo")
rich.print(f"[bold green]{'Loaded':>12}[/] {len(chemont)} terms from ChemOnt")

chemont_indices = { 
    term.id: i
    for i, term in enumerate(sorted(chemont.terms()))
    if term.id != "CHEMONTID:9999999"
}


# --- Load features and classes ----------------------------------------------

# load the whole MIBiG 3.1
features = anndata.read("data/datasets/mibig3.1/features.hdf5")
classes = anndata.read("data/datasets/mibig3.1/classes.hdf5")
assert (features.obs.index == classes.obs.index).all()

# --- Filter classes and features --------------------------------------------

# remove compounds with unknown structure
features = features[~classes.obs.unknown_structure]
classes = classes[~classes.obs.unknown_structure]

# compute weights for the hamming distance: superclasses have a higher weight than leaves
weights = classes.varp["subclasses"].sum(axis=1).A1

# remove classes with less than 10 members
mask = (classes.var.n_positives >= 10) & (classes.var.n_positives < classes.n_obs - 10)
weights = weights[mask]
classes = classes[:, mask]
rich.print(f"[bold green]{'Using':>12}[/] {classes.n_vars} target classes")

# remove features absent from training set
features = features[:, features.X.sum(axis=0).A1 > 0]
rich.print(f"[bold green]{'Using':>12}[/] {features.n_vars} features")

# extract feature names for speeding-up the HMMER query
features_names = {x:i for i,x in enumerate(features.var_names)}
class_names = {x:i for i,x in enumerate(classes.var_names)}


# --- Fit classifier ---------------------------------------------------------

rich.print(f"[bold blue]{'Training':>12}[/] classifier")
classifier = sklearn.multiclass.OneVsRestClassifier(
    sklearn.linear_model.LogisticRegression(penalty="l1", solver="liblinear"),
    n_jobs=-1,
)
classifier.fit(features.X, classes.X)
rich.print(f"[bold green]{'Finished':>12}[/] training classifier")

classifier.class_names_ = list(classes.var_names)
classifier.feature_names_ = list(features.var_names)

joblib.dump(classifier, "build/lr.dmp")







