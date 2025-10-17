import argparse
import os
import pathlib

import anndata
import matplotlib.pyplot as plt
import sklearn.metrics
import rich.progress

plt.rcParams['svg.fonttype'] = 'none'

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--classes", required=True)
parser.add_argument("--probas", required=True)
parser.add_argument("-o", "--output", required=True, type=pathlib.Path)
args = parser.parse_args()

classes = anndata.read_h5ad(args.classes)
probas = anndata.read_h5ad(args.probas)
classes = classes[probas.obs_names, probas.var_names]

plt.figure(1, figsize=(2, 2))

os.makedirs(args.output, exist_ok=True)
for i, class_name in enumerate(rich.progress.track(classes.var_names)):
    plt.figure(1)
    y_true = classes.obs_vector(class_name)#X[:, i].A.T[0]
    y_pred = probas.obs_vector(class_name) #X[:, i]
    pr, rc, _ = sklearn.metrics.precision_recall_curve(y_true, probas.X[:, i])
    aupr = sklearn.metrics.average_precision_score(y_true, probas.X[:, i])
    plt.plot(rc, pr, label=f"AUC=({aupr:.3f})", color="black")
    freq = classes.var.n_positives[class_name] / classes.n_obs
    plt.axhline(freq, linestyle="--", color="grey")
    plt.xticks([0, 0.5, 1])
    plt.yticks([0, 0.5, 1])
    #plt.legend()
    # plt.title(f"C{class_name.split(':', 1)[1]} ({classes.var.name[class_name]})")
    plt.savefig(os.path.join(args.output, f"{class_name}.svg"))
    plt.clf()
