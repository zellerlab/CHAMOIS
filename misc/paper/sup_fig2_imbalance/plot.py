import argparse
import os
import sys
import pathlib

import numpy
import anndata
import pandas
from matplotlib import rcParams
from matplotlib import pyplot as plt

# fix font embedding in SVG images
rcParams['svg.fonttype'] = 'none'

folder = pathlib.Path(__file__)
while not folder.joinpath("chamois").exists():
    folder = folder.parent
    
sys.path.insert(0, str(folder))
from chamois.predictor import ChemicalOntologyPredictor

t = ChemicalOntologyPredictor.trained()

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
parser.add_argument("-o", "--output", required=True)
parser.add_argument("--cv-report", required=True)
parser.add_argument("--show", action="store_true")
args = parser.parse_args()

classes = anndata.read_h5ad(args.input)
classes = classes[:, t.classes_.index]
classes = classes[~classes.obs.unknown_structure]
support = classes.X.sum(axis=0).A1

report = pandas.read_table(args.cv_report, index_col="class")
report = report.loc[t.classes_.index]

plt.figure(1, figsize=(6, 12))
plt.subplot(2, 1, 1)
plt.hist(support, bins=20)
plt.xlabel("Observations per class")
plt.ylabel("Number of classes")

plt.subplot(2, 1, 2)
plt.scatter(support, report["auprc"], alpha=0.5)
plt.xlabel("Observations per class")
plt.ylabel("Cross-validation AUPRC")
S = numpy.sort(support)
plt.plot(S, S/classes.n_obs, linestyle="--", color="gray")
plt.xscale('log')

plt.savefig(args.output)

if args.show:
    plt.show()



# parser = argparse.ArgumentParser()
# parser.add_argument("--output", required=True, type=pathlib.Path)
# args = parser.parse_args()


# print(f"{t.intercept_.shape=}, {t.coef_.shape=}")

# w = numpy.vstack([t.intercept_.T, t.coef_.toarray()])
# print(f"{w.shape=}")

# print(t.features_)
# weights = anndata.AnnData(X=w, obs=pandas.DataFrame(index=["Intercept"] + list(t.features_.index)), var=t.classes_).to_df()
# weights.to_csv(args.output, sep="\t")
