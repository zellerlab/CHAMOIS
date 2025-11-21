import collections
import contextlib
import itertools
import json
import math
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

from functions import clean_mols, get_ecfp6_fingerprints, get_tanimoto

# disable logging
RDLogger.DisableLog('rdApp.error')

# fix font embedding in SVG images
rcParams['svg.fonttype'] = 'none'

folder = pathlib.Path(__file__).absolute().parent
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
    row = {"Cluster": cluster}
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

# --- Add CHAMOIS predictions to the table -----------------------------------

rich.print(f"[bold blue]{'Adding':>12}[/] CHAMOIS results to the predictions")

chamois_predictions = []

for cluster, rows in predictions.groupby("Cluster"):

    # compute fingerprints for true SMILES
    true_smiles = rows["True SMILES"].unique()
    true_mols = clean_mols(true_smiles)
    true_fps = get_ecfp6_fingerprints(true_mols)

    # get predictions for current BGC
    # bgc_id, _ = os.path.splitext(cluster)
    bgc_id = cluster
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
predictions.sort_values(["Cluster", "Method"], inplace=True)
predictions.to_csv(folder.joinpath("predictions.tsv"), index=False, sep="\t")

# --- Summarize performance per method per BGC ---------------------------------

table = []
for cluster, rows in predictions.groupby("Cluster"):
    stats = {"Cluster": cluster, "True Smiles": rows["True SMILES"].values[0]}
    for method in ("NP.searcher", "PRISM 1", "PRISM 4", "antiSMASH 4", "CHAMOIS"):
        kmed = f"{method} (median)"
        kmax = f"{method} (max)"
        if (rows.Method == method).any():
            stats[kmed] = rows[rows.Method == method]["Tanimoto coefficient"].median()
            stats[kmax] = rows[rows.Method == method]["Tanimoto coefficient"].max()
        else:
            stats[kmed] = math.nan
            stats[kmax] = math.nan
    table.append(stats)

table = pandas.DataFrame(table)
table = pandas.merge(table, types, on="Cluster")
table.to_csv(folder.joinpath("medians.tsv"), index=False, sep="\t")
