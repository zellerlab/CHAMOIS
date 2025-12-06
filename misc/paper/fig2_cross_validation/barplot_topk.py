import argparse
import collections
import copy
import json
import os
import sys
import pathlib

import anndata
import pandas
import numpy
import rich.progress
import sklearn.metrics
from matplotlib import pyplot as plt

folder = pathlib.Path(__file__).parent
while not folder.joinpath("chamois").exists():
    folder = folder.parent
sys.path.insert(0, str(folder))

import chamois.predictor

plt.rcParams['svg.fonttype'] = 'none'
K = 100
TYPE_PALETTE = {
    'Polyketide': '#1e88e5',
    'NRP': '#884ea0',
    'RiPP': '#fdd835',
    'Saccharide': '#ec407a',
    'Terpene': '#009688',
    'Alkaloid': '#ef6c00',
    'Other': '#607d8b',
    'Mixed': '#80BA5A',
}

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--classes", required=True)
parser.add_argument("--types", required=True)
parser.add_argument("--probas", required=True)
parser.add_argument("-o", "--output", required=True, type=pathlib.Path)
args = parser.parse_args()


cv_probas = anndata.read_h5ad(args.probas)
classes = anndata.read_h5ad(args.classes)
classes = classes[cv_probas.obs_names, cv_probas.var_names].copy()

types = pandas.read_csv(args.types, sep="\t", header=None, names=["bgc_id", "type"], index_col="bgc_id")
types["type"] = types["type"].apply(lambda ty: "Mixed" if ";" in ty else ty)
ids = []
values = []
baseline = []

for i, class_name in enumerate(rich.progress.track(classes.var_names)):
    y_true = classes.obs_vector(class_name)
    y_pred = cv_probas.obs_vector(class_name)
    values.append(sklearn.metrics.average_precision_score(y_true, y_pred))
    baseline.append(classes.var["n_positives"].loc[class_name] / classes.n_obs)

classes.var['aupr'] = values
classes.var['baseline'] = baseline
classes.var['diff'] = classes.var['aupr'] - classes.var['baseline']
classes.obs["type"] = types["type"][classes.obs_names]

top = classes.var.sort_values('diff', ascending=False).head(K).sort_values('aupr', ascending=False)
X = list(range(len(top)))

colormap = plt.get_cmap('turbo')
colors = colormap( top['aupr'].copy() )

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 8), gridspec_kw={'height_ratios':[24,1,8]}, sharex=True)

counts = collections.defaultdict(lambda: numpy.zeros(K))
for i, name in enumerate(top.index):
    for k, v in classes.obs[classes.obs_vector(name)]["type"].value_counts().items():
        counts[k][i] = v
freqs = copy.deepcopy(counts)
for i, name in enumerate(top.index):
    total = sum(counts[k][i] for k in freqs)
    for k in freqs.keys():
        freqs[k][i] /= total

for i in range(len(top.index)):
    axes[1].text(X[i] - 0.4, 0, int(sum(counts[k][i] for k in freqs)), rotation=90)
axes[1].set_axis_off()
axes[1].set_ylabel("BGC\ninstances")

bottom = numpy.zeros(K)
for k in TYPE_PALETTE:
    axes[2].bar(X, freqs[k], bottom=bottom, color=TYPE_PALETTE[k], label=k)
    bottom += freqs[k]
#axes[2].legend()
axes[2].set_ylabel("BGC Type\nFraction")
axes[2].set_xlim(-1, len(top))
axes[2].set_xticks(X, labels=["C" + x[10:] for x in top.index], rotation=90)

axes[0].bar(X, top['aupr'].values, color=colors)
axes[0].bar(X, top.baseline, color="gray") #, marker='x', color='black')
axes[0].set_xlim(-1, len(top))
axes[0].set_ylabel("AUPRC")
# axes[2].tight_layout()

fig.tight_layout()
plt.savefig(args.output)
plt.show()
