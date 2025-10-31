import argparse
import os
import sys
import pathlib

import numpy
import anndata
import pandas

folder = pathlib.Path(__file__)
while not folder.joinpath("chamois").exists():
    folder = folder.parent

sys.path.insert(0, str(folder))
from chamois.predictor import ChemicalOntologyPredictor

t = ChemicalOntologyPredictor.trained()

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, type=pathlib.Path)
args = parser.parse_args()


print(f"{t.intercept_.shape=}, {t.coef_.shape=}")

w = numpy.vstack([t.intercept_.T, t.coef_.toarray()])
print(f"{w.shape=}")

print(t.features_)
weights = anndata.AnnData(X=w, obs=pandas.DataFrame(index=["Intercept"] + list(t.features_.index)), var=t.classes_).to_df()
weights.to_csv(args.output, sep="\t")
