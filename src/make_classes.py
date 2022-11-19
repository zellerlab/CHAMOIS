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
import pandas
import pronto
import numpy
import rich.progress
import scipy.sparse
from openbabel import pybel

# Disable warnings from OpenBabel
pybel.ob.obErrorLog.SetOutputLevel(0)

# get paths from command line
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
parser.add_argument("--atlas", required=True)
parser.add_argument("--chemont", required=True)
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()


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

def get_classyfire(inchikey):
    if inchikey in inchikey_index:
        npaid = inchikey_index[inchikey]["npaid"]
        compound.setdefault("database_id", []).append(f"npatlas:{npaid}")
        if np_atlas[npaid]["classyfire"] is not None:
            rich.print(f"[bold green]{'Using':>12}[/] NPAtlas classification ({npaid}) for compound {compound['compound']!r} of {bgc_id}")
            return np_atlas[npaid]["classyfire"]
    elif inchikey in cache:
        rich.print(f"[bold blue]{'Using':>12}[/] cached annotations for compound {compound['compound']!r} of {bgc_id}")
        return cache[inchikey]
    else:
        # otherwise use the ClassyFire website API
        try:
            rich.print(f"[bold blue]{'Querying':>12}[/] ClassyFire for compound {compound['compound']!r} of {bgc_id}")
            with urllib.request.urlopen(f"http://classyfire.wishartlab.com/entities/{inchikey}.json") as res:
                if res.code == 200:
                    data = json.load(res)
                    if "class" in data:
                        cache[inchikey] = data
                        return data
                    else:
                        return None
        except HTTPError:
            return None
        finally:
            time.sleep(0.1)

annotations = {}
for bgc_id, bgc_compounds in rich.progress.track(compounds.items(), description=f"[bold blue]{'Classifying':>12}[/]"):
    # get annotations for every compound of the BGC
    annotations[bgc_id] = []
    for compound in bgc_compounds:
        # ignore compounds without structure (should have gotten one already)
        if "chem_struct" not in compound:
            rich.print(f"[bold red]{'Skipping':>12}[/] {compound['compound']!r} compound of {bgc_id} with no structure")
            continue
        # try to use classyfire by inchi
        inchikey = pybel.readstring("smi", compound['chem_struct'].strip()).write("inchikey").strip()
        classyfire = get_classyfire(inchikey)
        if classyfire is None:
            rich.print(f"[bold red]{'Failed':>12}[/] to get ClassyFire annotations for {compound['compound']!r} compound of {bgc_id}")
            continue
        # record annotations
        annotations[bgc_id].append(classyfire)


# --- Binarize classes -------------------------------------------------------

unknown_structure = numpy.zeros(len(bgc_ids), dtype=numpy.bool_)
classes = numpy.zeros((len(compounds), len(chemont_indices)), dtype=numpy.bool_)

for bgc_id in rich.progress.track(annotations, description=f"[bold blue]{'Binarizing':>12}[/]"):
    # record if this BGC has no chemical annotation available
    if not annotations[bgc_id]:
        bgc_index = bgc_indices[bgc_id]
        unknown_structure[bgc_index] = True
        continue
    # get recursive superclasses from annotated classes
    for annotation in annotations[bgc_id]:
        # get all parents by traversing the ontology transitively
        direct_parents = pronto.TermSet({
            chemont[direct_parent["chemont_id"]] # type: ignore
            for direct_parent in itertools.chain(
                [annotation["kingdom"], annotation["superclass"], annotation["class"], annotation["subclass"], annotation["direct_parent"]],  
                annotation["intermediate_nodes"],
                annotation["alternative_parents"],
            )
            if direct_parent is not None
        })
        all_parents = direct_parents.superclasses().to_set()
        # set label flag is compound belong to a class
        bgc_index = bgc_indices[bgc_id]
        for parent in all_parents:
            if parent.id != "CHEMONTID:9999999":
                classes[bgc_index, chemont_indices[parent.id]] = True


# --- Create annotated data --------------------------------------------------

# generate annotated data
data = anndata.AnnData(
    dtype=numpy.bool_,
    X=scipy.sparse.csr_matrix(classes),
    obs=pandas.DataFrame(index=bgc_ids),
    var=pandas.DataFrame(index=list(chemont_indices)),
)

# save annotated data
os.makedirs(os.path.dirname(args.output), exist_ok=True)
data.write(args.output)
 

