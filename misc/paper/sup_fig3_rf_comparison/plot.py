import argparse
import os
import sys
import pathlib

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

t = ChemicalOntologyPredictor.trained()

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", required=True)
parser.add_argument("--cv-report", required=True)
parser.add_argument("--rf-report", required=True)
parser.add_argument("--show", action="store_true")
args = parser.parse_args()

cv_report = pandas.read_table(args.cv_report, index_col="class")
cv_report = cv_report.loc[t.classes_.index]
X = cv_report["auprc"].values

rf_report = pandas.read_table(args.rf_report, index_col="class")
rf_report = rf_report.loc[t.classes_.index]
Y = rf_report["auprc"].values

plt.scatter(cv_report.auprc, rf_report.auprc, alpha=0.5)

out = linregress(X, Y)
plt.axline((0, out.intercept), slope=out.slope, linestyle="--", color="gray", label=f"$r = {out.rvalue:.3f}$")
plt.legend()
plt.xlabel("Logistic Regression AUPRC")
plt.ylabel("Random Forest AUPRC")

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
