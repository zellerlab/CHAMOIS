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

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, type=pathlib.Path)
parser.add_argument("-o", "--output", required=True, type=pathlib.Path)
parser.add_argument("--show", action="store_true")
args = parser.parse_args()

with open(args.model, "rb") as f:
    t = ChemicalOntologyPredictor.load(f)

X = (t.coef_ >= 2.0).sum(axis=0).A1

plt.figure(figsize=(6, 6))
plt.hist(X, bins=numpy.arange(7), width=0.8)
plt.xticks(numpy.arange(6) + 0.4, labels=numpy.arange(6))
plt.xlabel("Domains with CHAMOIS weight >=2.0")
plt.ylabel("ChemOnt classes")
plt.savefig(args.output)

if args.show:
    plt.show()
