import collections
import contextlib
import itertools
import json
import functools
import operator
import tarfile
import os
import pathlib
import posixpath
import sys
import webbrowser
from random import choice

import anndata
import pandas
import numpy
import rich.progress
import sklearn.metrics
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib import rcParams
from scipy.spatial.distance import jensenshannon
from rdkit.Contrib.IFG.ifg import identify_functional_groups
from rdkit import RDLogger
from rdkit.DataStructs import TanimotoSimilarity
from palettable.cartocolors.qualitative import Bold_10

# disable logging
RDLogger.DisableLog('rdApp.error')

# fix font embedding in SVG images
rcParams['svg.fonttype'] = 'none'

PROJECT_FOLDER = pathlib.Path(__file__).absolute().parents[3]
PALETTE = {
    "PRISM 1": "#23395d",
    "PRISM 4": "#92b5d9",
    "PRISM 4 + NPAtlas": "#45baac",
    "NP.searcher": "#e5d89d",
    "antiSMASH 4": "#a96184",
    "CHAMOIS": "#8d4004",
}


# --- Load clusters with sequences -------------------------------------------

PRISM_FAMILY_TO_TYPE = {
    'nonribosomal_peptide': "nonribosomal peptide",
    'type_i_polyketide': "type 1 polyketide",
    'ribosomal': "RiPP",
    'type_ii_polyketide': "type 2 polyketide",
    'nucleoside': "nucleoside",
    'aminoglycoside': "aminoglycoside",
    'bisindole': "bisindole",
    'phosphonate': "phosphonate",
    'beta_lactam': "beta-lactam",
    'hapalindole': "isonitrile alkaloid",
    'cyclodipeptide': "cyclodipeptide",
    'aminocoumarin': "aminocoumarin",
    'antimetabolite': "antimetabolite",
    'lincosamide': "lincoside",
    # other subfamilies
    'butyrolactone': "other",
    'nis_synthase': "other",
    'bacteriocin': "RiPP",
    'melanin': "other",
    'null': "other",
    'ectoine': "other",
    'homoserine_lactone': "other",
    'phenazine': "other",
    'resorcinol': "other",
    'furan': "other",
    'phosphoglycolipid': "other",
    'aryl_polyene': "other"
}

PRISM_FAMILY_TO_MIBIG = {
    'nonribosomal_peptide': "NRP",
    'type_i_polyketide': "Polyketide",
    'ribosomal': "RiPP",
    'type_ii_polyketide': "Polyketide",
    'nucleoside': "Other",
    'aminoglycoside': "Other",
    'bisindole': "Other",
    'phosphonate': "Other",
    'beta_lactam': "Other",
    'hapalindole': "Other",
    'cyclodipeptide': "Other",
    'aminocoumarin': "Other",
    'antimetabolite': "Other",
    'lincosamide': "Other",
    # other subfamilies
    'butyrolactone': "Other",
    'nis_synthase': "Other",
    'bacteriocin': "RiPP",
    'melanin': "Other",
    'null': "Other",
    'ectoine': "Other",
    'homoserine_lactone': "Other",
    'phenazine': "Other",
    'resorcinol': "Other",
    'furan': "Other",
    'phosphoglycolipid': "Other",
    'aryl_polyene': "Polyketide", # fatty acid
}

type_counts = collections.Counter()
cluster_types = {}
mibig_types = {}

with contextlib.ExitStack() as ctx:
    progress = ctx.enter_context(rich.progress.Progress())
    reader = ctx.enter_context(progress.open(PROJECT_FOLDER.joinpath("data", "prism4", "BGCs.tar"), "rb", description=f"[bold blue]{'Reading':>12}[/]"))
    tar = ctx.enter_context(tarfile.open(fileobj=reader, mode="r"))
    for entry in tar:
        if entry.name.startswith("./json") and entry.name.endswith(".json"):
            name, _ = os.path.splitext(os.path.basename(entry.name.replace("-", "_")))
            with tar.extractfile(entry) as f:
                data = json.load(f)
            for cluster in data["prism_results"]["clusters"]:
                for family in cluster["family"]:
                    type_counts[family.lower()] += 1
            cluster_types[name] = {
                PRISM_FAMILY_TO_TYPE[family.lower()]
                for cluster in data["prism_results"]["clusters"]
                for family in cluster["family"]
                if family.lower() in PRISM_FAMILY_TO_TYPE
            }
            mibig_types[name] = {
                PRISM_FAMILY_TO_MIBIG[family.lower()]
                for cluster in data["prism_results"]["clusters"]
                for family in cluster["family"]
                if family.lower() in PRISM_FAMILY_TO_MIBIG 
            }

# --- Make indicator table of cluster types ----------------------------------

rows = []
for cluster, bgc_types in cluster_types.items():
    row = {"Cluster": cluster}
    for ty in PRISM_FAMILY_TO_TYPE.values():
        row[ty] = ty in bgc_types
    for ty in PRISM_FAMILY_TO_MIBIG.values():
        row[ty] = (ty in mibig_types[cluster]) | row.get(ty, False)
    rows.append(row)
types = pandas.DataFrame(rows).sort_values("Cluster")

# --- Load merged predictions ------------------------------------------------

predictions = pandas.read_table(PROJECT_FOLDER.joinpath("misc", "paper", "sup_fig5_prism4", "predictions.tsv"))
predictions = predictions[predictions["Method"].isin(PALETTE)]

# --- Select subset of predictions with all methods --------------------------

rich.print(f"[bold blue]{'Selecting':>12}[/] subset of predictions")

cluster_subset = functools.reduce(
    operator.and_,
    [set(predictions["Cluster"][predictions["Method"] == method]) for method in ("NP.searcher", "PRISM 1", "PRISM 4", "antiSMASH 4")]
)

subset_predictions = predictions[ predictions["Cluster"].isin(cluster_subset) ]

# --- Make combined figure ---------------------------------------------------

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# --- Plot summary Tanimoto by methods ---------------------------------------

medians = subset_predictions[["Method", "Cluster", "Tanimoto coefficient"]].groupby(["Method", "Cluster"]).median()

methods = []
boxes = []
coefs = []

for i, (method, rows) in enumerate(medians.reset_index().groupby("Method")):
    rows = rows[~rows["Tanimoto coefficient"].isna()]
    bp = ax1.boxplot([rows["Tanimoto coefficient"]], positions=[i], patch_artist=True)
    plt.setp(bp['boxes'], facecolor=PALETTE[method]) 
    plt.setp(bp['medians'], color='black')
    boxes.append(bp["boxes"][0])
    methods.append(method)
    coefs.append(rows["Tanimoto coefficient"])

i = methods.index("CHAMOIS")
offset = 0.1
for j, m in reversed(list(enumerate(methods))):
    if m != "CHAMOIS":
        p = scipy.stats.ttest_ind(coefs[i], coefs[j]).pvalue
        txt = "ns" if p > 0.05 else "*" if p > 0.01 else "**" if p > 0.001 else "***" if p > 0.0001 else "****" 
        ax1.text( (i + j) / 2, 1 + offset + 0.02, txt, ha="center")
        ax1.plot([i, i], [1 + offset, 1 + offset + 0.01], '-', color="black")
        ax1.plot([j, j], [1 + offset, 1 + offset + 0.01], '-', color="black")
        ax1.plot([i, j], [1 + offset + 0.01, 1 + offset + 0.01], '-', color="black")
        offset += 0.1

#ax1.legend(boxes, methods, loc='upper right')
ax1.set_xticks(range(len(methods)), labels=methods, rotation=45)
ax1.set_yticks(numpy.linspace(0, 1, 6))
ax1.set_ylim(bottom=-0.20)
ax1.set_ylabel("Jaccard Similarity")

# plt.tight_layout()
# plt.savefig(pathlib.Path(__file__).absolute().parent.joinpath("boxplot_by_method.png"))
# plt.savefig(pathlib.Path(__file__).absolute().parent.joinpath("boxplot_by_method.svg"))

# --- Detailed plot by MIBiG type --------------------------------------------

MIBIG_TYPES = [
    "Polyketide",
    "NRP",
    "RiPP",
    "Other",
]

# get predictions by type
subset = ["PRISM 4", "PRISM 4 + NPAtlas", "CHAMOIS"]
detailed_predictions = predictions[predictions["Method"].isin(set(subset))]
detailed_predictions = pandas.merge(detailed_predictions, types, how="left", on="Cluster")
detailed_predictions = detailed_predictions[~detailed_predictions["Tanimoto coefficient"].isna()]
cluster_subset = functools.reduce(operator.and_, [set(detailed_predictions["Cluster"][detailed_predictions["Method"] == m]) for m in subset])
detailed_predictions = detailed_predictions[detailed_predictions["Cluster"].isin(cluster_subset)]

# compute median of predictions per cluster per type
medians = {x: [] for x in subset}
for ty in MIBIG_TYPES:
    ty_predictions = detailed_predictions[detailed_predictions[ty]]
    for method in medians:
        ty_rows = ty_predictions[ty_predictions["Method"] == method]
        groups = ty_rows[["Cluster", "Tanimoto coefficient"]].groupby("Cluster", sort=True)
        medians[method].append(groups.median().reset_index()["Tanimoto coefficient"].values)

# render boxplot
X = numpy.arange(len(MIBIG_TYPES))
bplot3 = ax2.boxplot(medians["CHAMOIS"], positions=X-0.2, widths=0.1, patch_artist=True)
bplot1 = ax2.boxplot(medians["PRISM 4"], positions=X+0.0, widths=0.1, patch_artist=True)
bplot2 = ax2.boxplot(medians["PRISM 4 + NPAtlas"], positions=X+0.2, widths=0.1, patch_artist=True)
colors = [PALETTE[x] for x in subset]
for bplot, color in zip((bplot1, bplot2, bplot3), colors):
    plt.setp(bplot['boxes'], facecolor=color) 
    plt.setp(bplot['medians'], color='black')
    
ax2.legend([bplot["boxes"][0] for bplot in (bplot1, bplot2, bplot3)], subset, loc='upper left')

# show support values for classes
for x, m in zip(X, medians["CHAMOIS"]):
    ax2.text(x, -0.15, len(m), ha="center")
ax2.set_ylim(bottom=-0.20)

# show stats 
for x, m1, m2 in zip(X, medians["PRISM 4"], medians["CHAMOIS"]):
    p = scipy.stats.ttest_rel(m1, m2).pvalue
    txt = "ns" if p > 0.05 else "*" if p > 0.01 else "**" if p > 0.001 else "***" if p > 0.0001 else "****" 
    ax2.text(x-0.1, 1.12, txt, ha="center")
    ax2.plot([x-0.2, x-0.2], [1.10, 1.11], '-', color="black")
    ax2.plot([x+0.0, x+0.0], [1.10, 1.11], '-', color="black")
    ax2.plot([x-0.2, x+0.0], [1.11, 1.11], '-', color="black")
for x, m1, m2 in zip(X, medians["PRISM 4 + NPAtlas"], medians["CHAMOIS"]):
    p = scipy.stats.ttest_rel(m1, m2).pvalue
    txt = "ns" if p > 0.05 else "*" if p > 0.01 else "**" if p > 0.001 else "***" if p > 0.0001 else "****" 
    ax2.text(x, 1.22, txt, ha="center")
    ax2.plot([x-0.2, x-0.2], [1.20, 1.21], '-', color="black")
    ax2.plot([x+0.2, x+0.2], [1.20, 1.21], '-', color="black")
    ax2.plot([x-0.2, x+0.2], [1.21, 1.21], '-', color="black")
ax2.set_ylim(top=ax1.get_ylim()[1])

# show ticks
ax2.set_yticks(numpy.linspace(0, 1, 6))
ax2.set_xticks(X, labels=MIBIG_TYPES, rotation=45)
ax2.set_ylabel("Jaccard Similarity")


# --- BGCat Benchmark ----------------------------------------------------------

# Load ground truth
classes = anndata.read_h5ad(PROJECT_FOLDER.joinpath("data", "datasets", "native", "classes.npclassifier.hdf5"))
classes = classes[~classes.obs.unknown_structure]
classes = classes[:, ~classes.var.duplicated("name")]
classes.var.set_index("name", inplace=True)

# Load CHAMOIS predictions
chamois = anndata.read_h5ad(PROJECT_FOLDER.joinpath("misc", "paper", "sup_table8_benchmark_chamois_npclassifier", "chamois_predictions.hdf5"))
chamois = chamois[:, ~chamois.var.duplicated("name")]
chamois.var.set_index("name", inplace=True)

# Load BGCat predictions
bgcat = anndata.read_h5ad(PROJECT_FOLDER.joinpath("misc", "paper", "sup_table9_benchmark_bgcat_npclassifier", "bgcat_predictions.hdf5"))

# Get common obs names
obs_names = sorted(set(bgcat.obs_names) & set(classes.obs_names) & set(chamois.obs_names))

# Reindex observations
classes = classes[obs_names, :]
bgcat = bgcat[obs_names, :]
chamois = chamois[obs_names, :]

print(f"CHAMOIS classes: {chamois.n_vars}")
print(f"BGCat   classes: {bgcat.n_vars}")

# Micro precision-recall
ground_truth_chamois = classes[:, chamois.var_names].X.toarray().ravel()
preds_chamois = chamois.X.toarray().ravel()
auprc = sklearn.metrics.average_precision_score(ground_truth_chamois, preds_chamois)
pr, rc, _ = sklearn.metrics.precision_recall_curve(ground_truth_chamois, preds_chamois)
ax3.plot(rc, pr, label=f"CHAMOIS (AUPRC={auprc:5.3f})", color=PALETTE["CHAMOIS"])

# Compute precision-recall for default CHAMOIS (p=0.5)
pr = sklearn.metrics.precision_score(ground_truth_chamois, preds_chamois > 0.5)
rc = sklearn.metrics.recall_score(ground_truth_chamois, preds_chamois > 0.5)
ax3.scatter(rc, pr, marker="o", label=f"CHAMOIS / P=0.5 (F1={2*pr*rc / (pr + rc):5.3f})", edgecolors='black', color=PALETTE["CHAMOIS"])

# Micro precision-recall
ground_truth_bgcat = classes[:, bgcat.var_names].X.toarray().ravel()
preds_bgcat = bgcat.X.toarray().ravel()
auprc = sklearn.metrics.average_precision_score(ground_truth_bgcat, preds_bgcat)
pr, rc, _ = sklearn.metrics.precision_recall_curve(ground_truth_bgcat, preds_bgcat)
ax3.plot(rc, pr, label=f"BGCat (AUPRC={auprc:5.3f})", color=Bold_10.hex_colors[-2])

# F1 score
macro_f1_score_chamois = sklearn.metrics.f1_score(ground_truth_chamois, preds_chamois > 0.5, average="macro")
macro_f1_score_bgcat = sklearn.metrics.f1_score(ground_truth_bgcat, preds_bgcat > 0.5, average="macro")
print(f"Macro F1 (CHAMOIS): {macro_f1_score_chamois:5.3f}")
print(f"Macro F1 (BGCat):   {macro_f1_score_bgcat:5.3f}")

# Compute precision-recall for default BGCat (top20)
ground_truth_bgcat = classes[:, bgcat.var_names].X.toarray().ravel()
top20_bgcat = numpy.zeros((bgcat.n_obs, bgcat.n_vars), dtype=bool)
indices = numpy.argsort(bgcat.X.toarray(), axis=1)[:, -20:]
for i in range(indices.shape[0]):
    top20_bgcat[i, indices[i]] = True
pr = sklearn.metrics.precision_score(ground_truth_bgcat, top20_bgcat.ravel())
rc = sklearn.metrics.recall_score(ground_truth_bgcat, top20_bgcat.ravel())
ax3.scatter(rc, pr, marker="o", label=f"BGCat / Top20 (F1={2*pr*rc / (pr + rc):5.3f})", edgecolors="black", color=Bold_10.hex_colors[-2])

plt.legend()
plt.ylabel("Precision")
plt.xlabel("Recall")

plt.tight_layout()
plt.savefig(pathlib.Path(__file__).parent.joinpath("Fig5_combined.png"))
plt.savefig(pathlib.Path(__file__).parent.joinpath("Fig5_combined.svg"))
# plt.show()
