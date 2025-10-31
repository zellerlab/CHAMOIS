import argparse
import collections
import sys
import os
import pathlib
import gzip
import json
import urllib.request

import anndata
import fastobo
import numpy
import pandas
import rich.progress
import scipy.stats
from rich.console import Console

folder = pathlib.Path(__file__).parent

PROJECT_FOLDER = folder
while not PROJECT_FOLDER.joinpath("chamois").exists():
    PROJECT_FOLDER = PROJECT_FOLDER.parent




coordinates = pandas.read_table(PROJECT_FOLDER.joinpath("data", "datasets", "native", "coordinates.tsv"))
types = pandas.read_table(PROJECT_FOLDER.joinpath("data", "datasets", "native", "types.tsv"))
classes = anndata.read_h5ad(PROJECT_FOLDER.joinpath("data", "datasets", "native", "classes.hdf5"))

# print(coordinates)
# print(types)
# print(classes.obs)

table = pandas.merge(coordinates, types[["bgc_id", "type"]], on="bgc_id")
table = pandas.merge(table, classes.obs[["groups"]], left_on="bgc_id", right_index=True)

class_ids = []
for i, bgc_id in enumerate(classes.obs_names):
    c = classes.var_vector(bgc_id)
    ids = classes.var_names[c]
    class_ids.append((bgc_id, ";".join(ids)))


table = pandas.merge(table, pandas.DataFrame(class_ids, columns=["bgc_id", "classes"]), on="bgc_id")
table.to_csv(folder.joinpath("benchmark_dataset.tsv"), sep="\t", index=False)
