import argparse
import collections
import dataclasses
import itertools
import json
import gzip
import os
import sys
import urllib.request
import time
from urllib.error import HTTPError
from typing import Dict, List

import anndata
import disjoint_set
import joblib
import pandas
import pronto
import pubchempy
import numpy
import rdkit.Chem
import rich.progress
import scipy.sparse
from rdkit import RDLogger
from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..", "..")))
import chamois.classyfire

# disable logging
RDLogger.DisableLog('rdApp.warning')

# get paths from command line
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
parser.add_argument("--atlas", required=True)
parser.add_argument("--chemont", required=True)
parser.add_argument("-o", "--output", required=True)
parser.add_argument("--output-json")
parser.add_argument("-D", "--distance", type=float, default=0.5)
parser.add_argument("--cache")
parser.add_argument("--wishart", action="store_true", default=False)
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
chemont_indices = { term.id:i for i, term in enumerate(sorted(chemont.terms()))}


# --- Make adjacency matrix for the class graph ------------------------------

parents = scipy.sparse.dok_matrix((len(chemont_indices), len(chemont_indices)), dtype=numpy.bool_)
superclasses = scipy.sparse.dok_matrix((len(chemont_indices), len(chemont_indices)), dtype=numpy.bool_)
for term_id, i in chemont_indices.items():
    for superclass in chemont[term_id].superclasses():
        j = chemont_indices[superclass.id]
        superclasses[i, j] = True
    for superclass in chemont[term_id].superclasses(with_self=False, distance=1):
        j = chemont_indices[superclass.id]
        parents[i, j] = True   

children = scipy.sparse.dok_matrix((len(chemont_indices), len(chemont_indices)), dtype=numpy.bool_)
subclasses = scipy.sparse.dok_matrix((len(chemont_indices), len(chemont_indices)), dtype=numpy.bool_)
for term_id, i in chemont_indices.items():
    for subclass in chemont[term_id].subclasses():
        j = chemont_indices[subclass.id]
        subclasses[i, j] = True
    for subclass in chemont[term_id].subclasses(with_self=False, distance=1):
        j = chemont_indices[subclass.id]
        children[i, j] = True


# --- Get ClassyFire annotations for all compounds ----------------------------

@memory.cache
def get_compounds(cids):
    return pubchempy.get_compounds(cids)

cache = {}
if args.wishart:
    CLASSYFIRE_URL = "http://classyfire.wishartlab.com/entities/"
else:
    CLASSYFIRE_URL = "https://cfb.fiehnlab.ucdavis.edu/entities/"

annotations = {}
classyfire_client = chamois.classyfire.Client(entities_url=CLASSYFIRE_URL)
for bgc_id, bgc_compounds in rich.progress.track(compounds.items(), description=f"[bold blue]{'Classifying':>12}[/]"):
    # get annotations for every compound of the BGC
    annotations[bgc_id] = []
    for compound in bgc_compounds:
        # ignore compounds without structure (should have gotten one already)
        if "chem_struct" not in compound:
            annotations[bgc_id].append(None)
            rich.print(f"[bold yellow]{'Skipping':>12}[/] {compound['compound']!r} compound of [purple]{bgc_id}[/] with no structure")
            continue
        # use InChi key to find annotation in NPAtlas
        inchikey = rdkit.Chem.inchi.MolToInchiKey(rdkit.Chem.MolFromSmiles(compound['chem_struct'].strip()))
        if inchikey in inchikey_index:
            npaid = inchikey_index[inchikey]["npaid"]
            compound.setdefault("database_id", []).append(f"npatlas:{npaid}")
            if np_atlas[npaid]["classyfire"] is not None:
                rich.print(f"[bold green]{'Found':>12}[/] NPAtlas classification ([bold cyan]{npaid}[/]) for compound {compound['compound']!r} of [purple]{bgc_id}[/]")
                classyfire_client.cache[inchikey] = np_atlas[npaid]["classyfire"]
                annotations[bgc_id].append(chamois.classyfire.Classification.from_dict(np_atlas[npaid]["classyfire"]))
                continue
        # try to use classyfire by InChi key othewrise
        rich.print(f"[bold blue]{'Fetching':>12}[/] ClassyFire annotations for compound {compound['compound']!r} of [purple]{bgc_id}[/]")
        try:
            classyfire = classyfire_client.fetch(inchikey)
        except (RuntimeError, HTTPError) as err:
            rich.print(f"[bold red]{'Failed':>12}[/] to get ClassyFire annotations for {compound['compound']!r} compound of [purple]{bgc_id}[/]: {err}")
        else:
            rich.print(f"[bold green]{'Retrieved':>12}[/] ClassyFire annotations for {compound['compound']!r} compound of [purple]{bgc_id}[/]")
            annotations[bgc_id].append(classyfire)
            continue
        #
        pubchem_id = next((db_id.split(":")[1] for db_id in compound.get("database_id", []) if db_id.startswith("pubchem")), None)
        if pubchem_id is not None:
            c = get_compounds([pubchem_id])[0]
            try:
                classyfire = classyfire_client.fetch(c.inchikey)
            except (RuntimeError, HTTPError) as err:
                rich.print(f"[bold red]{'Failed':>12}[/] to get ClassyFire annotations for {compound['compound']!r} compound of [purple]{bgc_id}[/]: {err}")
            else:
                rich.print(f"[bold green]{'Retrieved':>12}[/] ClassyFire annotations for {compound['compound']!r} compound of [purple]{bgc_id}[/]")
                annotations[bgc_id].append(classyfire)
                continue

        #try:
        #    rich.print(f"[bold blue]{'Sending':>12}[/] ClassyFire query {compound['compound']!r} compound of [purple]{bgc_id}[/]")
        #    query = classyfire_client.query([ compound['chem_struct'].strip() ])
        #    status = "In Queue"
        #    while status == "In Queue" or status == "Processing":
        #        time.sleep(10)
        #        status = query.status
        #        rich.print(f"[bold blue]{'Waiting':>12}[/] for ClassyFire to finish (query {query.id!r})")
        #        break
        #    if status != "Done":
        #        raise RuntimeError("Classyfire failed")
        #except (RuntimeError, HTTPError) as err:
        #    rich.print(f"[bold red]{'Failed':>12}[/] ClassyFire annotation of {compound['compound']!r} compound of [purple]{bgc_id}[/]")
        #    annotations[bgc_id].append(None)
        #else:
        #    rich.print(f"[bold green]{'Finished':>12}[/] ClassyFire annotation of {compound['compound']!r} compound of [purple]{bgc_id}[/]")
        #    out = classyfire_client.retrieve(query)
        #    if out["invalid_entities"]:
        #        rich.print(f"[bold red]{'Failed':>12}[/] to get ClassyFire annotations for {compound['compound']!r} compound of [purple]{bgc_id}[/]")
        #    else:
        #        classyfire = chamois.classyfire.Classification.from_dict(out['entities'][0])
        #        annotations[bgc_id].append(classyfire)
        #    continue
        annotations[bgc_id].append(None)

# --- Export classification to JSON if requiested ---------------------------

if args.output_json:
    class Encoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, chamois.classyfire.Classification):
                return dataclasses.asdict(o)
            return super().default(o)

    annotations_with_info = {
        bgc_id: [
            {"annotations": annotations[bgc_id][i], "compound": compounds[bgc_id][i]}
            for i in range(len(compounds[bgc_id]))
        ]
        for bgc_id in annotations
    }
    with open(args.output_json, "w") as f:
        json.dump(annotations_with_info, f, cls=Encoder, sort_keys=True, indent=4)


# --- Binarize classes -------------------------------------------------------

def full_classification(classification):
    return pronto.TermSet({chemont[t.id] for t in classification.terms}).superclasses().to_set()

unknown_structure = numpy.zeros(len(bgc_ids), dtype=numpy.bool_)
classes = numpy.zeros((len(compounds), len(chemont_indices)), dtype=numpy.bool_)
smiles = [""]*len(compounds)
names = [""]*len(compounds)
inchikey = [""] * len(compounds)

total = 0
for annotation_compounds in annotations.values():
    total += sum(map(bool, annotation_compounds))
rich.print(f"[bold blue]{'Retrieved':>12}[/] {total} compounds in {len(annotations)} BGCs")


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
    # record compount structure
    bgc_compound = compounds[bgc_id][best_index]
    bgc_annotation = annotations[bgc_id][best_index]
    smiles[bgc_index] = bgc_annotation.smiles
    names[bgc_index] = bgc_compound["compound"]
    inchikey[bgc_index] = bgc_annotation.inchikey.split("=", 1)[1]
    # record classification and metadata for compound
    for parent in full_classification(bgc_annotation):
        classes[bgc_index, chemont_indices[parent.id]] = True


# --- Build groups using MHFP6 distances -------------------------------------

rich.print(f"[bold blue]{'Building':>12}[/] MHFP6 fingerprints for {len(bgc_indices)} compounds")

encoder = MHFPEncoder(2048, 42)
group_set = disjoint_set.DisjointSet({ i:i for i in range(len(bgc_ids)) })
fps = numpy.array(encoder.EncodeSmilesBulk(smiles, kekulize=True))
indices = itertools.combinations(range(len(bgc_ids)), 2)
total = len(bgc_ids) * (len(bgc_ids) - 1) / 2

for (i, j) in rich.progress.track(indices, total=total, description=f"[bold blue]{'Joining':>12}[/]"):
    if not unknown_structure[i] and not unknown_structure[j]:
        d = scipy.spatial.distance.hamming(fps[i], fps[j])
        if d < args.distance:
            group_set.union(i, j)

n = sum(1 for _ in group_set.itersets())
rich.print(f"[bold green]{'Built':>12}[/] {n} groups of molecules with MHFP6 distance over {args.distance}")


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
            inchikey=inchikey,
            groups=[group_set[i] for i in range(len(bgc_ids))]
        ),
    ),
    var=pandas.DataFrame(
        index=list(chemont_indices),
        data=dict(
            name=[chemont[id_].name for id_ in chemont_indices],
            description=[chemont[id_].definition for id_ in chemont_indices],
            n_positives=classes.sum(axis=0)
        )
    ),
    varp=dict(
        parents=parents.tocsr(),
        children=children.tocsr(),
        subclasses=subclasses.tocsr(),
        superclasses=superclasses.tocsr(),
    )
)

# save annotated data
os.makedirs(os.path.dirname(args.output), exist_ok=True)
data.write(args.output)


