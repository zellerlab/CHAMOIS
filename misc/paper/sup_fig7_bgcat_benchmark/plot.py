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

plt.figure(1, figsize=(6,6))

# Micro precision-recall
ground_truth_chamois = classes[:, chamois.var_names].X.toarray().ravel()
preds_chamois = chamois.X.toarray().ravel()
auprc = sklearn.metrics.average_precision_score(ground_truth_chamois, preds_chamois)
pr, rc, _ = sklearn.metrics.precision_recall_curve(ground_truth_chamois, preds_chamois)
plt.plot(rc, pr, label=f"CHAMOIS (AUPRC={auprc:5.3f})", color=Bold_10.hex_colors[0])

# Micro precision-recall
ground_truth_bgcat = classes[:, bgcat.var_names].X.toarray().ravel()
preds_bgcat = bgcat.X.toarray().ravel()
auprc = sklearn.metrics.average_precision_score(ground_truth_bgcat, preds_bgcat)
pr, rc, _ = sklearn.metrics.precision_recall_curve(ground_truth_bgcat, preds_bgcat)
plt.plot(rc, pr, label=f"BGCat (AUPRC={auprc:5.3f})", color=Bold_10.hex_colors[1])


# F1 score
macro_f1_score_chamois = sklearn.metrics.f1_score(ground_truth_chamois, preds_chamois > 0.5, average="macro")
macro_f1_score_bgcat = sklearn.metrics.f1_score(ground_truth_bgcat, preds_bgcat > 0.5, average="macro")
print(f"Macro F1 (CHAMOIS): {macro_f1_score_chamois:5.3f}")
print(f"Macro F1 (BGCat):   {macro_f1_score_bgcat:5.3f}")


plt.legend()
plt.ylabel("Precision")
plt.xlabel("Recall")

plt.tight_layout()
plt.savefig(pathlib.Path(__file__).parent.joinpath("pr.png"))
plt.savefig(pathlib.Path(__file__).parent.joinpath("pr.svg"))
plt.show()
