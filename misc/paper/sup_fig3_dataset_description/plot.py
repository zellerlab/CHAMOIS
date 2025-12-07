import argparse
import os
import sys
import pathlib

import anndata
import numpy
import pandas
from scipy.stats import linregress
from matplotlib import rcParams
from matplotlib import pyplot as plt

# fix font embedding in SVG images
rcParams['svg.fonttype'] = 'none'

folder = pathlib.Path(__file__)
while not folder.joinpath("chamois").exists():
    folder = folder.parent
    
sys.path.insert(0, str(folder))
from chamois.predictor import ChemicalOntologyPredictor

TYPE_PALETTE = {
    'Polyketide': '#1e88e5',
    'NRP': '#884ea0',
    'RiPP': '#fdd835',
    'Saccharide': '#ec407a',
    'Terpene': '#009688',
    'Alkaloid': '#ef6c00',
    'Other': '#607d8b',
    'Mixed': '#80BA5A',
    'Unknown': '#bebebe',
}

t = ChemicalOntologyPredictor.trained()

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--features", required=True)
parser.add_argument("-c", "--classes", required=True)
parser.add_argument("-t", "--types", required=True)
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()

# load classes and features
classes = anndata.read_h5ad(args.classes)
features = anndata.read_h5ad(args.features)
features = features[classes.obs_names, :]

features = features[:, t.features_.index]

# load types and apply "Mixed"
types = pandas.read_table(args.types, header=None, index_col=0, names=["type"])
types["type"] = types["type"].apply(lambda x: "Mixed" if ";" in x else x)

# compute number of domains per BGC
n_domains = features.X.sum(axis=1).A1

# 
data = pandas.DataFrame(data=n_domains, index=classes.obs_names, columns=["n_domains"])
data = pandas.merge(data, types, left_index=True, right_index=True)

# 
_, bins = numpy.histogram(n_domains, bins=20)
w = bins[1] - bins[0]

# plot histogram
plt.figure(1, figsize=(6, 6))

bottom = numpy.zeros(bins.shape[0] - 1)
for ty in TYPE_PALETTE.keys():
    rows = data[data["type"] == ty]
    h, _ = numpy.histogram(rows["n_domains"], bins=bins)
    p = plt.bar(bins[:h.shape[0]] + w/2, h, bottom=bottom, width=4.0, color=TYPE_PALETTE[ty], label=ty)
    bottom += h

plt.legend()
plt.ylim(0, bottom.max() + 10)
plt.xlabel("Number of Domains")
plt.ylabel("Number of BGCs")
plt.savefig(args.output)
plt.show()