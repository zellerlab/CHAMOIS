import json
import pathlib

import anndata
import pandas
import numpy
import sklearn.metrics

import matplotlib.pyplot as plt
from palettable.cartocolors.qualitative import Bold_10

folder = pathlib.Path(__file__).absolute()
while not folder.joinpath("chamois").is_dir():
    folder = folder.parent

# Load ground truth
classes = anndata.read_h5ad(folder.joinpath("data", "datasets", "native", "classes.npclassifier.hdf5"))
classes = classes[~classes.obs.unknown_structure]
classes = classes[:, ~classes.var.duplicated("name")]
classes.var.set_index("name", inplace=True)

# Load CHAMOIS predictions
chamois = anndata.read_h5ad(folder.joinpath("misc", "paper", "sup_table8_benchmark_chamois_npclassifier", "chamois_predictions.hdf5"))
chamois = chamois[:, ~chamois.var.duplicated("name")]
chamois.var.set_index("name", inplace=True)

# Load BGCat predictions
bgcat = anndata.read_h5ad(folder.joinpath("misc", "paper", "sup_table9_benchmark_bgcat_npclassifier", "bgcat_predictions.hdf5"))

# Get common obs names
obs_names = sorted(set(bgcat.obs_names) & set(classes.obs_names) & set(chamois.obs_names))

# Reindex observations
classes = classes[obs_names, :]
bgcat = bgcat[obs_names, :]
chamois = chamois[obs_names, :]

print(f"CHAMOIS classes: {chamois.n_vars}")
print(f"BGCat   classes: {bgcat.n_vars}")

ground_truth_chamois = classes[:, chamois.var_names]
ground_truth_bgcat = classes[:, bgcat.var_names]

preds_chamois = chamois.X.toarray() > 0.5
preds_bgcat = bgcat.X.toarray() > 0.5

rows = []

for obs_name in obs_names:

    y_chamois = chamois.var_vector(obs_name) > 0.5
    y_bgcat = bgcat.var_vector(obs_name) > 0.5

    truth_chamois = ground_truth_chamois.var_vector(obs_name)
    truth_bgcat = ground_truth_bgcat.var_vector(obs_name)

    tp_chamois = (truth_chamois & y_chamois).sum()
    tp_bgcat = (truth_bgcat & y_bgcat).sum()

    fp_chamois = (~truth_chamois & y_chamois).sum()
    fp_bgcat = (~truth_bgcat & y_bgcat).sum()

    fn_chamois = (truth_chamois & ~y_chamois).sum()
    fn_bgcat = (truth_bgcat & ~y_bgcat).sum()

    tn_chamois = (~truth_chamois & ~y_chamois).sum()
    tn_bgcat = (~truth_bgcat & ~y_bgcat).sum()

    pr_chamois = (tp_chamois) / (tp_chamois + fp_chamois)
    pr_bgcat = (tp_bgcat) / (tp_bgcat + fp_bgcat)

    rc_chamois = (tp_chamois) / (tp_chamois + fn_chamois)
    rc_bgcat = (tp_bgcat) / (tp_bgcat + fn_bgcat)

    f1_chamois = 2*(tp_chamois) / (2*tp_chamois + fp_chamois + fn_chamois)
    f1_bgcat = 2*(tp_bgcat) / (2*tp_bgcat + fp_bgcat + fn_bgcat)

    rows.append(dict(

        bgc_id=obs_name,
        compound_name=chamois.obs.loc[obs_name, "compound"],

        tp_chamois=tp_chamois,
        fp_chamois=fp_chamois,
        fn_chamois=fn_chamois,
        tn_chamois=tn_chamois,
        pr_chamois=pr_chamois,
        rc_chamois=rc_chamois,
        f1_chamois=f1_chamois,

        tp_bgcat=tp_bgcat,
        fp_bgcat=fp_bgcat,
        fn_bgcat=fn_bgcat,
        tn_bgcat=tn_bgcat,
        pr_bgcat=pr_bgcat,
        rc_bgcat=rc_bgcat,
        f1_bgcat=f1_bgcat
    ))


for avg in ["micro", "macro", "samples"]:
    rows.append(dict(
        id=f"{avg}_average",

        pr_chamois=sklearn.metrics.precision_score(ground_truth_chamois.X.toarray(), preds_chamois, average=avg, zero_division=1),
        rc_chamois=sklearn.metrics.recall_score(ground_truth_chamois.X.toarray(), preds_chamois, average=avg, zero_division=0),
        f1_chamois=sklearn.metrics.f1_score(ground_truth_chamois.X.toarray(), preds_chamois, average=avg),

        pr_bgcat=sklearn.metrics.precision_score(ground_truth_bgcat.X.toarray(), preds_bgcat, average=avg, zero_division=1),
        rc_bgcat=sklearn.metrics.recall_score(ground_truth_bgcat.X.toarray(), preds_bgcat, average=avg, zero_division=0),
        f1_bgcat=sklearn.metrics.f1_score(ground_truth_bgcat.X.toarray(), preds_bgcat, average=avg),
    ))


df = pandas.DataFrame(rows).set_index("id")
df.to_csv(pathlib.Path(__file__).parent.joinpath("summary.tsv"), sep="\t")
