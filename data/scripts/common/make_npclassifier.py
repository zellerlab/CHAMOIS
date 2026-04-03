import argparse
import collections
import dataclasses
import itertools
import json
import gzip
import os
import pathlib
import sys
import shutil
import urllib.request
import urllib.parse
import time
from urllib.error import HTTPError
from typing import Dict, List

import anndata
import disjoint_set
import pandas
import pronto
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
parser.add_argument("--index", required=True)
parser.add_argument("-o", "--output", required=True)
parser.add_argument("-D", "--distance", type=float, default=0.5)
# parser.add_argument("--cache")
args = parser.parse_args()

# create persistent cache
# if args.cache:
#     rich.print(f"[bold green]{'Using':>12}[/] joblib cache folder {args.cache!r}")
#     os.makedirs(args.cache, exist_ok=True)

import platformdirs
cache = pathlib.Path(platformdirs.user_cache_dir('CHAMOIS', 'ZellerLab')).joinpath("npclassifier")

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


# --- Load the NP-Classifier index --------------------------------------------

with rich.progress.open(args.index, "rb", description=f"[bold blue]{'Loading':>12}[/]") as handle:
    index = json.load(handle)

np_pathways_index = {k:int(v) for k,v in index["Pathway"].items()}
np_classes_index = {k:int(v) for k,v in index["Class"].items()}
np_superclasses_index = {k:int(v) for k,v in index["Superclass"].items()}

np_pathways = sorted(np_pathways_index, key=np_pathways_index.get)
np_classes = sorted(np_classes_index, key=np_classes_index.get)
np_superclasses = sorted(np_superclasses_index, key=np_superclasses_index.get)

np = [*[ "Pathway_"+x for x in np_pathways], *["Class_"+x for x in np_classes], *["Superclass_"+x for x in np_superclasses]]
np_index = {k:i for i,k in enumerate(np)}

# --- Make adjacency matrix for the class graph -------------------------------

parents = scipy.sparse.dok_matrix((len(np), len(np)), dtype=numpy.bool_)
children = scipy.sparse.dok_matrix((len(np), len(np)), dtype=numpy.bool_)

for class_name in np_classes:
    class_key = np_classes_index[class_name]
    for parent_key in index["Class_hierarchy"][str(class_key)]["Superclass"]:
        parent_name = np_superclasses[parent_key]
        assert np_superclasses_index[parent_name] == parent_key
        i = np_index["Class_"+class_name]
        j = np_index["Superclass_"+parent_name]
        parents[i, j] = True
        children[j, i] = True

for superclass_name in np_superclasses:
    class_key = np_superclasses_index[superclass_name]
    if str(class_key) not in index["Super_hierarchy"]:
        continue
    for parent_key in index["Super_hierarchy"][str(class_key)]["Pathway"]:
        parent_name = np_pathways[parent_key]
        assert np_pathways_index[parent_name] == parent_key
        i = np_index["Superclass_"+superclass_name]
        j = np_index["Pathway_"+parent_name]
        parents[i, j] = True
        children[j, i] = True


# --- Get NP Classifier annotations for all compounds --------------------------

@dataclasses.dataclass
class Annotation:
    classes: dataclasses.field(default_factory=list)
    superclasses: dataclasses.field(default_factory=list)
    pathways: dataclasses.field(default_factory=list)
    smiles: str

    @property
    def n_terms(self):
        return len(self.classes) + len(self.superclasses) + len(self.pathways)
    
def full_classification(d):
    return [*d["class_results"], *d["superclass_results"], *d["pathway_results"]]

annotations = {}
for bgc_id, bgc_compounds in rich.progress.track(compounds.items(), description=f"[bold blue]{'Classifying':>12}[/]"):
    # get annotations for every compound of the BGC
    annotations[bgc_id] = []
    for compound in bgc_compounds:
        # ignore compounds without structure (should have gotten one already)
        if "chem_struct" not in compound:
            annotations[bgc_id].append(None)
            rich.print(f"[bold yellow]{'Skipping':>12}[/] {compound['compound']!r} compound of [purple]{bgc_id}[/] with no structure")
            continue

        # # use InChi key to find annotation in NPAtlas
        inchikey = rdkit.Chem.inchi.MolToInchiKey(rdkit.Chem.MolFromSmiles(compound['chem_struct'].strip()))
        # if inchikey in inchikey_index:
        #     npaid = inchikey_index[inchikey]["npaid"]
        #     compound.setdefault("database_id", []).append(f"npatlas:{npaid}")
        #     if np_atlas[npaid]["npclassifier"] is not None:
        #         rich.print(f"[bold green]{'Found':>12}[/] NPAtlas classification ([bold cyan]{npaid}[/]) for compound {compound['compound']!r} of [purple]{bgc_id}[/]")
        #         # classyfire_client.cache[inchikey] = np_atlas[npaid]["classyfire"]
        #         classification = full_classification(np_atlas[npaid]["npclassifier"])
        #         if classification:
        #             annotations[bgc_id].append(Annotation(classification, compound['chem_struct'].strip()))
        #         continue

        # query the NPClassifier server by SMILES
        cache_path = cache.joinpath(f"{inchikey}.json.gz")
        if not cache_path.exists():
            rich.print(f"[bold blue]{'Sending':>12}[/] NP Classifier query {compound['compound']!r} compound of [purple]{bgc_id}[/]")
            url = "https://npclassifier.gnps2.org/classify?smiles={}".format(urllib.parse.quote(compound["chem_struct"]))
            with urllib.request.urlopen(url) as res:
                with gzip.open(cache_path, "wb") as dst:
                    shutil.copyfileobj(res, dst)
        with gzip.open(cache_path, mode="rb") as src:
            data = json.load(src)
        if full_classification(data):
            rich.print(f"[bold green]{'Retrieved':>12}[/] NP Classifier annotations for {compound['compound']!r} compound of [purple]{bgc_id}[/]")
            annotations[bgc_id].append(Annotation(data["class_results"], data['superclass_results'], data["pathway_results"], compound['chem_struct'].strip()))
            continue
            
        # # try to use classyfire by InChi key othewrise
        # rich.print(f"[bold blue]{'Fetching':>12}[/] ClassyFire annotations for compound {compound['compound']!r} of [purple]{bgc_id}[/]")
        # try:
        #     classyfire = classyfire_client.fetch(inchikey)
        # except (RuntimeError, HTTPError) as err:
        #     rich.print(f"[bold red]{'Failed':>12}[/] to get ClassyFire annotations for {compound['compound']!r} compound of [purple]{bgc_id}[/]")
        # else:
        #     rich.print(f"[bold green]{'Retrieved':>12}[/] ClassyFire annotations for {compound['compound']!r} compound of {bgc_id}")
        #     annotations[bgc_id].append(classyfire)
        #     continue
        # try:
        #     rich.print(f"[bold blue]{'Sending':>12}[/] ClassyFire query {compound['compound']!r} compound of {bgc_id}")
        #     query = classyfire_client.query([ compound['chem_struct'].strip() ])
        #     status = "In Queue"
        #     while status == "In Queue" or status == "Processing":
        #         time.sleep(10)
        #         status = query.status
        #     if status != "Done":
        #         raise RuntimeError("Classyfire failed")
        # except (RuntimeError, HTTPError) as err:
        #     rich.print(f"[bold red]{'Failed':>12}[/] ClassyFire annotation of {compound['compound']!r} compound of [purple]{bgc_id}[/]")
        #     annotations[bgc_id].append(None)
        # else:
        #     rich.print(f"[bold green]{'Finished':>12}[/] ClassyFire annotation of {compound['compound']!r} compound of {bgc_id}")
        #     classyfire = chamois.classyfire.Classification.from_dict(classyfire_client.retrieve(query)['entities'][0])
        #     annotations[bgc_id].append(classyfire)
        #     continue


# --- Binarize classes -------------------------------------------------------

unknown_structure = numpy.zeros(len(bgc_ids), dtype=numpy.bool_)
classes = numpy.zeros((len(compounds), len(np_index)), dtype=numpy.bool_)
smiles = [""]*len(compounds)
names = [""]*len(compounds)
inchikey = [""] * len(compounds)

for bgc_id in rich.progress.track(annotations, description=f"[bold blue]{'Binarizing':>12}[/]"):
    bgc_index = bgc_indices[bgc_id]
    # record if this BGC has no chemical annotation available
    if not any(annotations[bgc_id]):
        unknown_structure[bgc_index] = True
        if compounds[bgc_id]:
            names[bgc_index] = compounds[bgc_id][0]["compound"]
        continue
    # find compound with the most classes
    
    sorted_indices = sorted(
        range(len(annotations[bgc_id])),
        key=lambda i: -1 if not annotations[bgc_id][i] else annotations[bgc_id][i].n_terms,
        reverse=True,
    )
    for best_index in sorted_indices:
        # record compound structure
        bgc_compound = compounds[bgc_id][best_index]
        bgc_annotation = annotations[bgc_id][best_index]
        smiles[bgc_index] = bgc_annotation.smiles
        names[bgc_index] = bgc_compound["compound"]
        # inchikey[bgc_index] = bgc_annotation.inchikey.split("=", 1)[1]
        # record classification and metadata for compound
        try:
            for c in bgc_annotation.classes:
                classes[bgc_index, np_index["Class_"+c]] = True
            for c in bgc_annotation.superclasses:
                classes[bgc_index, np_index["Superclass_"+c]] = True
            for c in bgc_annotation.pathways:
                classes[bgc_index, np_index["Pathway_"+c]] = True
        except KeyError as err:
            rich.print(f"[bold yellow]{'Warning':>12}[/] unknown class name: {err}")
            classes[bgc_index, :] = False
        else:
            break
    else:
        unknown_structure[bgc_index] = True

# --- Build groups using MHFP6 distances -------------------------------------

rich.print(f"[bold blue]{'Building':>12}[/] MHFP6 fingerprints for {len(bgc_indices)} compounds")

encoder = MHFPEncoder(2048, 42)
group_set = disjoint_set.DisjointSet({ i:i for i in range(len(bgc_ids)) })
fps = numpy.array(encoder.EncodeSmilesBulk(smiles, kekulize=True))
indices = itertools.combinations(range(len(bgc_ids)), 2)
total = len(bgc_ids) * (len(bgc_ids) - 1) / 2

for (i, j) in rich.progress.track(indices, total=total, description=f"[bold blue]{'Grouping':>12}[/]"):
    if not unknown_structure[i] and not unknown_structure[j]:
        d = scipy.spatial.distance.hamming(fps[i], fps[j])
        if d < args.distance:
            group_set.union(i, j)

n = sum(1 for _ in group_set.itersets())
rich.print(f"[bold green]{'Built':>12}[/] {n} groups of molecules with MHFP6 distance over {args.distance}")


# --- Create annotated data --------------------------------------------------

# generate annotated data
data = anndata.AnnData(
    X=scipy.sparse.csr_matrix(classes, dtype=bool),
    obs=pandas.DataFrame(
        index=bgc_ids,
        data=dict(
            unknown_structure=unknown_structure,
            compound=names,
            smiles=smiles,
            # inchikey=inchikey,
            groups=[group_set[i] for i in range(len(bgc_ids))]
        ),
    ),
    var=pandas.DataFrame(
        index=[f"NP:{i:03}" for i in np_index.values()],
        data=dict(
            name=[x.split("_", maxsplit=1)[1] for x in np_index],
            rank=[x.split("_", maxsplit=1)[0] for x in np_index],
            # description=[chemont[id_].definition for id_ in chemont_indices],
            n_positives=classes.sum(axis=0)
        )
    ),
    varp=dict(
        parents=parents.tocsr(),
        children=children.tocsr(),
    #     subclasses=subclasses.tocsr(),
    #     superclasses=superclasses.tocsr(),
    )
)

# save annotated data
os.makedirs(os.path.dirname(args.output), exist_ok=True)
data.write(args.output)


