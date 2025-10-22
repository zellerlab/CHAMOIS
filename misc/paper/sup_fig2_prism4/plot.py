import collections
import contextlib
import itertools
import json
import functools
import operator
import tarfile
import os
import pathlib
import sys
import webbrowser
from random import choice

import anndata
import pandas
import numpy
import rich.progress
import matplotlib.pyplot as plt
import sklearn.metrics
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib import rcParams
from scipy.spatial.distance import jensenshannon
from rdkit.Contrib.IFG.ifg import identify_functional_groups
from rdkit import RDLogger

from functions import clean_mols, get_ecfp6_fingerprints, get_tanimoto

# disable logging
RDLogger.DisableLog('rdApp.error')

# fix font embedding in SVG images
rcParams['svg.fonttype'] = 'none'

PROJECT_FOLDER = pathlib.Path(__file__).absolute().parents[3]
PALETTE = {
    "PRISM 1": "#23395d",
    "PRISM 4": "#92b5d9",
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
    row = {"Cluster": os.path.splitext(cluster)[0].replace("-", "_")}
    for ty in PRISM_FAMILY_TO_TYPE.values():
        row[ty] = ty in bgc_types
    for ty in PRISM_FAMILY_TO_MIBIG.values():
        row[ty] = (ty in mibig_types[cluster]) | row.get(ty, False)
    rows.append(row)
types = pandas.DataFrame(rows).sort_values("Cluster")

# --- Load NPAtlas -----------------------------------------------------------

rich.print(f"[bold blue]{'Loading':>12}[/] Natural Product Atlas")
npatlas = anndata.read_h5ad(PROJECT_FOLDER.joinpath("data", "npatlas", "classes.hdf5"))


# --- Load CHAMOIS search results --------------------------------------------

rich.print(f"[bold blue]{'Loading':>12}[/] CHAMOIS search results")
search_results = pandas.read_table(pathlib.Path(__file__).absolute().parent.joinpath("search_results.tsv"))
search_results = search_results[search_results["rank"] == 1]


# --- Load PRISM4 predictions ------------------------------------------------

rich.print(f"[bold blue]{'Loading':>12}[/] PRISM4 predictions")
predictions = pandas.read_excel(PROJECT_FOLDER.joinpath("data", "prism4", "predictions.xlsx"))
predictions = predictions[~predictions["Predicted SMILES"].isna()]
predictions["Cluster"] = predictions["Cluster"].str.replace("-", "_").str.rsplit(".", n=1).str[0]

# --- Add CHAMOIS predictions to the table -----------------------------------

rich.print(f"[bold blue]{'Adding':>12}[/] CHAMOIS results to the predictions")

chamois_predictions = []

for cluster, rows in predictions.groupby("Cluster"):

    # compute fingerprints for true SMILES
    true_smiles = rows["True SMILES"].unique()
    true_mols = clean_mols(true_smiles)
    true_fps = get_ecfp6_fingerprints(true_mols)

    # get predictions for current BGC
    bgc_id, _ = os.path.splitext(cluster)
    chamois_hits = search_results[search_results.bgc_id == bgc_id]

    # extract SMILES from predictions 
    pred_smiles = [npatlas.obs.loc[hit.index].smiles for hit in chamois_hits.itertuples()]
    pred_mols = clean_mols(pred_smiles)
    pred_fps = get_ecfp6_fingerprints(pred_mols)

    # compute Tanimoto similarity
    tcs = get_tanimoto(true_fps, pred_fps)

    # add the CHAMOIS predictions to the table
    # (same code as `calculate-tanimoto-coefficients.py`)
    true_col = [y for x in pred_smiles for y in true_smiles]
    pred_col = [x for x in pred_smiles for y in true_smiles]
    res = pandas.DataFrame({'Cluster': cluster, 
                        'True SMILES': true_col,
                        'Predicted SMILES': pred_col,
                        'Tanimoto coefficient': tcs, 
                        'Method': 'CHAMOIS' })
    chamois_predictions.append(res)

predictions = pandas.concat(itertools.chain([predictions], chamois_predictions), ignore_index=True)

# for method, rows in predictions.groupby("Method"):
#     print(method, rows["Cluster"].unique())

# --- Select subset of predictions with all methods --------------------------

rich.print(f"[bold blue]{'Selecting':>12}[/] subset of predictions")

cluster_subset = functools.reduce(
    operator.and_,
    [set(predictions["Cluster"][predictions["Method"] == method]) for method in ("NP.searcher", "PRISM 1", "PRISM 4", "antiSMASH 4")]
)

subset_predictions = predictions[ predictions["Cluster"].isin(cluster_subset) ]


# --- Plot summary Tanimoto by methods ---------------------------------------

fig, ax1 = plt.subplots(nrows=1, ncols=1)
medians = subset_predictions[["Method", "Cluster", "Tanimoto coefficient"]].groupby(["Method", "Cluster"]).median()

methods = []
boxes = []
for i, (method, rows) in enumerate(medians.reset_index().groupby("Method")):
    rows = rows[~rows["Tanimoto coefficient"].isna()]
    bp = ax1.boxplot([rows["Tanimoto coefficient"]], positions=[i], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor(PALETTE[method])
    boxes.append(bp["boxes"][0])
    methods.append(method)

ax1.legend(boxes, methods, loc='upper right')
ax1.set_xticks( range(len(methods)), labels=methods )

plt.tight_layout()
plt.savefig(pathlib.Path(__file__).absolute().parent.joinpath("boxplot_by_method.png"))
plt.savefig(pathlib.Path(__file__).absolute().parent.joinpath("boxplot_by_method.svg"))

# --- Detailed plot by cluster type (Median + Max) ---------------------------

TYPES = [
    "aminocoumarin", 
    "aminoglycoside", 
    "antimetabolite", 
    "beta-lactam", 
    "bisindole", 
    "cyclodipeptide", 
    "isonitrile alkaloid", 
    "lincoside", 
    "nonribosomal peptide", 
    "nucleoside", 
    "other", 
    "phosphonate", 
    "RiPP", 
    "type 1 polyketide", 
    "type 2 polyketide"
]

detailed_predictions = predictions[(predictions["Method"] == "PRISM 4") | (predictions["Method"] == "CHAMOIS")]
detailed_predictions = pandas.merge(detailed_predictions, types, how="left", on="Cluster")
detailed_predictions = detailed_predictions[~detailed_predictions["Tanimoto coefficient"].isna()]

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 6))
for (method, ax) in zip(["PRISM 4", "CHAMOIS" ], [ax1, ax2]):
        
    X = numpy.arange(len(TYPES))
    maximums = []
    medians = []

    for ty in TYPES:
        ty_rows = detailed_predictions[detailed_predictions[ty] & (detailed_predictions["Method"] == method)]
        groups = ty_rows[["Cluster", "Tanimoto coefficient"]].groupby("Cluster")
        medians.append(groups.median().reset_index()["Tanimoto coefficient"].values)
        maximums.append(groups.max().reset_index()["Tanimoto coefficient"].values)

    bp1 = ax.boxplot(medians, positions=X-0.15, widths=0.2, patch_artist=True)
    bp2 = ax.boxplot(maximums, positions=X+0.15, widths=0.2, patch_artist=True)
    ax.set_xticks(X, labels=TYPES, rotation=45)
    ax.set_title(method)

    # set color of boxes
    for patch in bp1["boxes"]:
        patch.set_facecolor("#43aa9e")
    for patch in bp2["boxes"]:
        patch.set_facecolor("#a94599")

    # show legend for boxes
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Median', 'Maximum'], loc='upper left')
    
    # show support values for classes
    for x, m in zip(X, medians):
        ax.text(x, -0.15, len(m), ha="center")
    ax.set_ylim(bottom=-0.20)


plt.tight_layout()
plt.savefig(pathlib.Path(__file__).absolute().parent.joinpath("boxplot_by_type.prism_figure.png"))
plt.savefig(pathlib.Path(__file__).absolute().parent.joinpath("boxplot_by_type.prism_figure.svg"))


# --- Detailed plot by cluster type (CHAMOIS to PRISM comparison) ------------

TYPES = [
    "aminocoumarin", 
    "aminoglycoside", 
    "antimetabolite", 
    "beta-lactam", 
    "bisindole", 
    "cyclodipeptide", 
    "isonitrile alkaloid", 
    "lincoside", 
    "nonribosomal peptide", 
    "nucleoside", 
    "other", 
    "phosphonate", 
    "RiPP", 
    "type 1 polyketide", 
    "type 2 polyketide"
]

# get predictions by type
detailed_predictions = predictions[(predictions["Method"] == "PRISM 4") | (predictions["Method"] == "CHAMOIS")]
detailed_predictions = pandas.merge(detailed_predictions, types, how="left", on="Cluster")
detailed_predictions = detailed_predictions[~detailed_predictions["Tanimoto coefficient"].isna()]
cluster_subset = functools.reduce(operator.and_, [set(detailed_predictions["Cluster"][detailed_predictions["Method"] == m]) for m in ("PRISM 4", "CHAMOIS")])
detailed_predictions = detailed_predictions[detailed_predictions["Cluster"].isin(cluster_subset)]

# compute median of predictions per cluster per type
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(12, 3))
medians = {"PRISM 4": [], "CHAMOIS": []}
for ty in TYPES:
    ty_predictions = detailed_predictions[detailed_predictions[ty]]
    for method in ["PRISM 4", "CHAMOIS"]:
        ty_rows = ty_predictions[ty_predictions["Method"] == method]
        groups = ty_rows[["Cluster", "Tanimoto coefficient"]].groupby("Cluster", sort=True)
        medians[method].append(groups.median().reset_index()["Tanimoto coefficient"].values)
    # print(ty, len(medians["PRISM 4"][-1]), len(medians["CHAMOIS"][-1]))

# render boxplot
X = numpy.arange(len(TYPES))
bplot1 = ax.boxplot(medians["PRISM 4"], positions=X-0.15, widths=0.2, patch_artist=True)
bplot2 = ax.boxplot(medians["CHAMOIS"], positions=X+0.15, widths=0.2, patch_artist=True)
colors = [PALETTE["PRISM 4"], PALETTE["CHAMOIS"]]
for bplot, color in zip((bplot1, bplot2), colors):
    for patch in bplot['boxes']:
        patch.set_facecolor(color)
ax.legend([bplot1["boxes"][0], bplot2["boxes"][0]], ['PRISM 4', 'CHAMOIS'], loc='upper left')

# show support values for classes
for x, m in zip(X, medians["CHAMOIS"]):
    ax.text(x, -0.15, len(m), ha="center")
ax.set_ylim(bottom=-0.20)

# show stats 
for x, m1, m2 in zip(X, medians["PRISM 4"], medians["CHAMOIS"]):
    p = scipy.stats.ttest_rel(m1, m2).pvalue
    txt = "ns" if p > 0.05 else "*" if p > 0.01 else "**" if p > 0.001 else "***" if p > 0.0001 else "****" 
    ax.text(x, 1.05, txt, ha="center")
    ax.plot([x-0.15, x-0.15], [1.03, 1.04], '-', color="black")
    ax.plot([x+0.15, x+0.15], [1.03, 1.04], '-', color="black")
    ax.plot([x-0.15, x+0.15], [1.04, 1.04], '-', color="black")
ax.set_ylim(top=1.20)

# show ticks
ax.set_xticks(X, labels=TYPES, rotation=45)
# plt.tight_layout()
plt.savefig(pathlib.Path(__file__).absolute().parent.joinpath("boxplot_by_type.median_comparison.png"))
plt.savefig(pathlib.Path(__file__).absolute().parent.joinpath("boxplot_by_type.median_comparison.svg"))

# plt.show()


# --- Detailed plot by MIBiG type --------------------------------------------

MIBIG_TYPES = [
    "Polyketide",
    "NRP",
    "RiPP",
    "Other",
]

# get predictions by type
detailed_predictions = predictions[(predictions["Method"] == "PRISM 4") | (predictions["Method"] == "CHAMOIS") | (predictions["Method"] == "DIAMOND")]
detailed_predictions = pandas.merge(detailed_predictions, types, how="left", on="Cluster")
detailed_predictions = detailed_predictions[~detailed_predictions["Tanimoto coefficient"].isna()]
cluster_subset = functools.reduce(operator.and_, [set(detailed_predictions["Cluster"][detailed_predictions["Method"] == m]) for m in ("PRISM 4", "CHAMOIS")])
detailed_predictions = detailed_predictions[detailed_predictions["Cluster"].isin(cluster_subset)]

# compute median of predictions per cluster per type
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(6, 6))
medians = {"PRISM 4": [], "CHAMOIS": []}
for ty in MIBIG_TYPES:
    ty_predictions = detailed_predictions[detailed_predictions[ty]]
    for method in medians:
        ty_rows = ty_predictions[ty_predictions["Method"] == method]
        groups = ty_rows[["Cluster", "Tanimoto coefficient"]].groupby("Cluster", sort=True)
        medians[method].append(groups.median().reset_index()["Tanimoto coefficient"].values)

# render boxplot
X = numpy.arange(len(MIBIG_TYPES))
bplot1 = ax.boxplot(medians["PRISM 4"], positions=X-0.2, widths=0.2, patch_artist=True)
bplot2 = ax.boxplot(medians["CHAMOIS"], positions=X+0.00, widths=0.2, patch_artist=True)
colors = [PALETTE["PRISM 4"], PALETTE["CHAMOIS"]]
for bplot, color in zip((bplot1, bplot2), colors):
    for patch in bplot['boxes']:
        patch.set_facecolor(color)
ax.legend([bplot1["boxes"][0], bplot2["boxes"][0]], ['PRISM 4', 'CHAMOIS'], loc='upper left')

# show support values for classes
for x, m in zip(X, medians["CHAMOIS"]):
    ax.text(x, -0.15, len(m), ha="center")
ax.set_ylim(bottom=-0.20)

# show stats 
for x, m1, m2 in zip(X, medians["PRISM 4"], medians["CHAMOIS"]):
    p = scipy.stats.ttest_rel(m1, m2).pvalue
    txt = "ns" if p > 0.05 else "*" if p > 0.01 else "**" if p > 0.001 else "***" if p > 0.0001 else "****" 
    ax.text(x - 0.1, 1.05, txt, ha="center")
    ax.plot([x-0.2, x-0.2], [1.03, 1.04], '-', color="black")
    ax.plot([x+0.0, x+0.0], [1.03, 1.04], '-', color="black")
    ax.plot([x-0.2, x+0.0], [1.04, 1.04], '-', color="black")
ax.set_ylim(top=1.20)

# show ticks
ax.set_xticks(X, labels=MIBIG_TYPES, rotation=45)
# plt.tight_layout()
plt.savefig(pathlib.Path(__file__).absolute().parent.joinpath("boxplot_by_mibig.median_comparison.png"))
plt.savefig(pathlib.Path(__file__).absolute().parent.joinpath("boxplot_by_mibig.median_comparison.svg"))



plt.show()
