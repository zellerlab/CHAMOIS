import argparse
import collections
import itertools
import json
import gzip
import os
import urllib.request
import time
from urllib.error import HTTPError
from typing import Dict, List

import anndata
import joblib
import pandas
import pronto
import numpy
import rdkit.Chem
import rich.progress
import scipy.sparse
from rdkit import RDLogger

# disable logging
RDLogger.DisableLog('rdApp.warning')

# get paths from command line
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
parser.add_argument("--atlas", required=True)
parser.add_argument("--chemont", required=True)
parser.add_argument("-o", "--output", required=True)
parser.add_argument("--cache")
args = parser.parse_args()

# create persistent cache
if args.cache:
    rich.print(f"[bold green]{'Using':>12}[/] joblib cache folder {args.cache!r}")
    os.makedirs(args.cache, exist_ok=True)
memory = joblib.Memory(location=args.cache, verbose=False)

# --- Load MIBiG -------------------------------------------------------------

with rich.progress.open(args.input, "rb", description=f"[bold blue]{'Loading':>12}[/] BGCs") as handle:
    compounds = json.load(handle)

bgc_ids = sorted(compounds)
bgc_indices = {name:i for i, name in enumerate(bgc_ids)}

rich.print(f"[bold green]{'Loaded':>12}[/] {len(bgc_indices)} BGCs")


# --- Load NPAtlas -----------------------------------------------------------

with rich.progress.open(args.atlas, "rb", description=f"[bold blue]{'Loading':>12}[/] NPAtlas") as handle:
    data = json.load(gzip.open(handle))

np_atlas = {entry["npaid"]: entry for entry in data}
inchikey_index = {entry["inchikey"]: entry for entry in data}


# --- Load ChemOnt ------------------------------------------------------------

chemont = pronto.Ontology(args.chemont)
rich.print(f"[bold green]{'Loaded':>12}[/] {len(chemont)} terms from ChemOnt")

chemont_indices = {
    term.id: i
    for i, term in enumerate(sorted(chemont.terms()))
    if term.id != "CHEMONTID:9999999"
}

# --- Get ClassyFire annotations for all compounds ----------------------------

cache = {}

@memory.cache
def get_classyfire_inchikey(inchikey):
    # otherwise use the ClassyFire website API
    with urllib.request.urlopen(f"http://classyfire.wishartlab.com/entities/{inchikey}.json") as res:
        data = json.load(res)
        time.sleep(0.1)
        if "class" not in data:
            raise RuntimeError("classification not found")
        return data if "class" in data else None

annotations = {}
for bgc_id, bgc_compounds in rich.progress.track(compounds.items(), description=f"[bold blue]{'Classifying':>12}[/]"):
    # get annotations for every compound of the BGC
    annotations[bgc_id] = []
    for compound in bgc_compounds:
        # ignore compounds without structure (should have gotten one already)
        if "chem_struct" not in compound:
            annotations[bgc_id].append(None)
            rich.print(f"[bold yellow]{'Skipping':>12}[/] {compound['compound']!r} compound of {bgc_id} with no structure")
            continue
        # use InChi key to find annotation in NPAtlas
        inchikey = rdkit.Chem.inchi.MolToInchiKey(rdkit.Chem.MolFromSmiles(compound['chem_struct'].strip()))
        if inchikey in inchikey_index:
            npaid = inchikey_index[inchikey]["npaid"]
            compound.setdefault("database_id", []).append(f"npatlas:{npaid}")
            if np_atlas[npaid]["classyfire"] is not None:
                rich.print(f"[bold green]{'Found':>12}[/] NPAtlas classification ({npaid}) for compound {compound['compound']!r} of {bgc_id}")
                annotations[bgc_id].append(np_atlas[npaid]["classyfire"])
                continue
        # try to use classyfire by InChi key othewrise
        rich.print(f"[bold blue]{'Querying':>12}[/] ClassyFire for compound {compound['compound']!r} of {bgc_id}")
        try:
            classyfire = get_classyfire_inchikey(inchikey)
        except (RuntimeError, HTTPError):
            rich.print(f"[bold red]{'Failed':>12}[/] to get ClassyFire annotations for {compound['compound']!r} compound of {bgc_id}")
            annotations[bgc_id].append(None)
        else:
            rich.print(f"[bold green]{'Downloaded':>12}[/] ClassyFire annotations for {compound['compound']!r} compound of {bgc_id}")
            annotations[bgc_id].append(classyfire)
            continue


# --- Binarize classes -------------------------------------------------------

def full_classification(annotation):
    return pronto.TermSet({
        chemont[direct_parent["chemont_id"]] # type: ignore
        for direct_parent in itertools.chain(
            [annotation["kingdom"], annotation["superclass"], annotation["class"], annotation["subclass"], annotation["direct_parent"]],
            annotation["intermediate_nodes"],
            annotation["alternative_parents"],
        )
        if direct_parent is not None
    }).superclasses().to_set()

unknown_structure = numpy.zeros(len(bgc_ids), dtype=numpy.bool_)
classes = numpy.zeros((len(compounds), len(chemont_indices)), dtype=numpy.bool_)
smiles = [""]*len(compounds)
names = [""]*len(compounds)

for bgc_id in rich.progress.track(annotations, description=f"[bold blue]{'Binarizing':>12}[/]"):
    bgc_index = bgc_indices[bgc_id]
    # record if this BGC has no chemical annotation available
    if not any(annotations[bgc_id]):
        unknown_structure[bgc_index] = True
        if compounds[bgc_id]:
            names[bgc_index] = compounds[bgc_id][0]["compound"]
        continue
    # find compound with the most classes
    best_index = max(
        range(len(annotations[bgc_id])),
        key=lambda i: -1 if not annotations[bgc_id][i] else len(full_classification(annotations[bgc_id][i])),
    )
    assert best_index >= 0
    # record classification and metadata for compound
    bgc_compound = compounds[bgc_id][best_index]
    bgc_annotation = annotations[bgc_id][best_index]
    smiles[bgc_index] = bgc_annotation["smiles"]
    names[bgc_index] = bgc_compound["compound"]
    for parent in full_classification(bgc_annotation):
        if parent.id != "CHEMONTID:9999999":
            classes[bgc_index, chemont_indices[parent.id]] = True


# --- Make adjacency matrix for the class graph ------------------------------

superclasses = scipy.sparse.dok_matrix((len(chemont_indices), len(chemont_indices)), dtype=numpy.bool_)
for term_id, i in chemont_indices.items():
    for superclass in chemont[term_id].superclasses():
        if superclass.id != "CHEMONTID:9999999":
            j = chemont_indices[superclass.id]
            superclasses[i, j] = True

subclasses = scipy.sparse.dok_matrix((len(chemont_indices), len(chemont_indices)), dtype=numpy.bool_)
for term_id, i in chemont_indices.items():
    for subclass in chemont[term_id].subclasses():
        if subclass.id != "CHEMONTID:9999999":
            j = chemont_indices[subclass.id]
            subclasses[i, j] = True


# --- Create annotated data --------------------------------------------------

# generate annotated data
data = anndata.AnnData(
    dtype=numpy.bool_,
    X=scipy.sparse.csr_matrix(classes),
    obs=pandas.DataFrame(
        index=bgc_ids,
        data=dict(
            unknown_structure=unknown_structure,
            compound=names,
            smiles=smiles,
        ),
    ),
    var=pandas.DataFrame(
        index=list(chemont_indices),
        data=dict(
            name=[chemont[id_].name for id_ in chemont_indices],
            n_positives=classes.sum(axis=0)
        )
    ),
    varp=dict(
        subclasses=subclasses.tocsr(),
        superclasses=superclasses.tocsr(),
    )
)

# save annotated data
os.makedirs(os.path.dirname(args.output), exist_ok=True)
data.write(args.output)


