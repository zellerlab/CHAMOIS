import pathlib
import json

import anndata
import pandas
import numpy

import matplotlib.pyplot as plt
from palettable.cartocolors.qualitative import Bold_10

folder = pathlib.Path(__file__).absolute()
while not folder.joinpath("chamois").is_dir():
    folder = folder.parent

# Load BGcat predictions
with folder.joinpath("data", "datasets", "native", "bgcat", "native.probas_all_genes.json").open() as f:
    bgcat_predictions = json.load(f)

# Binarize BGcat labels
labels = bgcat_predictions["labels"]
names = []
probas = numpy.zeros((len(bgcat_predictions["bgcs"]), len(labels)))

for i, (name, preds) in enumerate(bgcat_predictions["bgcs"].items()):
    names.append(name)
    probas[i, :] = preds

predictions = anndata.AnnData(probas, obs=pandas.DataFrame(index=names), var=pandas.DataFrame(index=labels))
predictions.write_h5ad(pathlib.Path(__file__).parent.joinpath("bgcat_predictions.hdf5"))
predictions.to_df().to_csv(pathlib.Path(__file__).parent.joinpath("bgcat_predictions.tsv"), sep="\t")

