import pathlib

import anndata
import pandas

folder = pathlib.Path(__file__).absolute().parent
probas = anndata.read_h5ad(folder.joinpath("chamois_predictions.hdf5"))

table = pandas.DataFrame(data=probas.X, index=probas.obs_names, columns=probas.var_names)
table.columns = probas.var["name"]
table = pandas.concat([probas.obs, table], axis=1)
table.to_csv(folder.joinpath("chamois_predictions.tsv"), sep="\t")

