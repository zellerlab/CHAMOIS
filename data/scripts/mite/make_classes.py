import argparse
import urllib.error
import urllib.request
import json
import gzip
import pathlib
import itertools
import io
import time
import os
import sys
from urllib.error import HTTPError

import anndata
import Bio.Entrez
import Bio.SeqIO
import gb_io
import rdkit.Chem
import rich.progress
import pandas
import pronto
import numpy
import scipy.sparse
from rich.console import Console
from rdkit import RDLogger

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..", "..")))

import chamois.classyfire

# disable logging
RDLogger.DisableLog('rdApp.warning')

# get paths from command line
parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", required=True, type=pathlib.Path)
parser.add_argument("--output", "-o", required=True, type=pathlib.Path)
parser.add_argument("--jobs", "-j", type=int, default=0)
parser.add_argument("--chemont", required=True)
parser.add_argument("--atlas", required=True)
parser.add_argument("-D", "--distance", type=float, default=0.5)
parser.add_argument("--wishart", action="store_true", default=False)
args = parser.parse_args()

# make rich console
console = Console()

def smiles_to_inchikey(smiles):
    return rdkit.Chem.inchi.MolToInchiKey(rdkit.Chem.MolFromSmiles(smiles))

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

# --- Load MITE entries -------------------------------------------------------

with args.input.open() as src:
    entries = json.load(src)

# --- Get ClassyFire annotations for all compounds ----------------------------

if args.wishart:
    CLASSYFIRE_URL = "http://classyfire.wishartlab.com/entities/"
else:
    CLASSYFIRE_URL = "https://cfb.fiehnlab.ucdavis.edu/entities/"

annotations = {}
fails = []
classyfire_client = chamois.classyfire.Client(entities_url=CLASSYFIRE_URL)
cache = classyfire_client.cache

for entry in rich.progress.track(entries, description=f"[bold blue]{'Classifying':>12}[/]"):
    mite_id = entry["accession"]
    annotations[mite_id] = []
    reaction = entry["reactions"][0]["reactions"][0]
    product = reaction["products"][0]
    substrate = reaction["substrate"]

    # rich.print(f"[bold blue]{'Working':>12}[/] on [purple]{mite_id}[/]")
    # rich.print(f"[bold blue]{'Found':>12}[/] subsrate {substrate!r}")
    # rich.print(f"[bold blue]{'Found':>12}[/] product {product!r}")

    for name, compound in {"product": product, "substrate": substrate}.items():
        # fix wildcard compounds using a single carbon atom
        if "*" in compound:
            compound = compound.replace("*", "C")
        inchikey = rdkit.Chem.inchi.MolToInchiKey(rdkit.Chem.MolFromSmiles(compound))
        # use InChi key to find annotation in NPAtlas
        if inchikey in inchikey_index:
            npaid = inchikey_index[inchikey]["npaid"]
            if np_atlas[npaid]["classyfire"] is not None:
                rich.print(f"[bold green]{'Found':>12}[/] NPAtlas classification ([bold cyan]{npaid}[/]) for {name} of [purple]{mite_id}[/]")
                classyfire_client.cache[inchikey] = np_atlas[npaid]["classyfire"]
                annotations[mite_id].append(chamois.classyfire.Classification.from_dict(np_atlas[npaid]["classyfire"]))
                continue
        if inchikey in classyfire_client.cache:
            rich.print(f"[bold green]{'Found':>12}[/] cached ClassyFire annotations for {name} of [purple]{mite_id}[/]")
            annotations[mite_id].append(chamois.classyfire.Classification.from_dict(classyfire_client.cache[inchikey]))
            continue
        annotations[mite_id].append(None)
        continue
        # try to use classyfire by InChi key othewrise
        rich.print(f"[bold blue]{'Fetching':>12}[/] ClassyFire annotations for {name} {inchikey!r} of [purple]{mite_id}[/]")
        try:
            classyfire = classyfire_client.fetch(inchikey)
        except (RuntimeError, HTTPError) as err:
            rich.print(f"[bold red]{'Failed':>12}[/] to get ClassyFire annotations for {name} of [purple]{mite_id}[/]")
        else:
            rich.print(f"[bold green]{'Retrieved':>12}[/] ClassyFire annotations for {name} of [purple]{mite_id}[/]")
            annotations[mite_id].append(classyfire)
            continue
        try:
            rich.print(f"[bold blue]{'Sending':>12}[/] ClassyFire query {name} compound of [purple]{mite_id}[/]")
            query = classyfire_client.query([ compound ])
            status = "In Queue"
            while status == "In Queue" or status == "Processing":
                time.sleep(10)
                status = query.status
            if status != "Done":
                raise RuntimeError("Classyfire failed")
        except (RuntimeError, HTTPError) as err:
            rich.print(f"[bold red]{'Failed':>12}[/] ClassyFire annotation of {name} compound of [purple]{mite_id}[/]")
            annotations[mite_id].append(None)
        else:
            rich.print(f"[bold green]{'Finished':>12}[/] ClassyFire annotation of {name} compound of [purple]{mite_id}[/]")
            classyfire = chamois.classyfire.Classification.from_dict(classyfire_client.retrieve(query)['entities'][0])
            annotations[mite_id].append(classyfire)
            continue

with open("fails.txt", "w") as f:
    f.writelines([ line + "\n" for line in fails ])

# --- Binarize classes -------------------------------------------------------

def full_classification(classification):
    return pronto.TermSet({chemont[t.id] for t in classification.terms}).superclasses().to_set()

def minimize(term_set: pronto.TermSet):
    t = pronto.TermSet(term_set)
    for term in term_set:
        for sup in term.superclasses(with_self=False):
            if sup in t:
                t.remove(sup)
    return t

mite_ids = sorted(entry["accession"] for entry in entries)
mite_index = { x:i for i, x in enumerate(mite_ids) }

unknown_structure = numpy.zeros(len(mite_ids), dtype=numpy.bool_)
classes = numpy.zeros((len(entries), len(chemont_indices)), dtype=numpy.bool_)

for entry in rich.progress.track(entries, description=f"[bold blue]{'Binarizing':>12}[/]"):    
    mite_id = entry["accession"]
    product_annotation, substrate_annotation = annotations[mite_id]
    if not substrate_annotation or not product_annotation:
        if not product_annotation:
            rich.print(f"[bold red]{'Failed':>12}[/] getting annotation for [purple]{mite_id}[/]: missing product annotation")
        elif not substrate_annotation:
            rich.print(f"[bold red]{'Failed':>12}[/] getting annotation for [purple]{mite_id}[/]: missing substrate annotation")
        unknown_structure[mite_index[mite_id]] = True
        continue

    substrate_terms = full_classification(substrate_annotation)
    product_terms = full_classification(product_annotation)

    for term in product_terms - substrate_terms:
        classes[mite_index[mite_id], chemont_indices[term.id]] = True

    rich.print(f"[bold green]{'Found':>12}[/] minimum annotation for [purple]{mite_id}[/]")
    #for t in minimize(product_terms - substrate_terms):
    #    rich.print(f"{' - ':>12} [bold cyan]{t.id}[/] ({t.name!r})")



# --- Create annotated data --------------------------------------------------

# generate annotated data
data = anndata.AnnData(
    X=scipy.sparse.csr_matrix(classes, dtype=numpy.bool_),
    obs=pandas.DataFrame(
        index=mite_ids,
        data=dict(
            unknown_structure=unknown_structure,
            # compound=names,
            # smiles=smiles,
            # inchikey=inchikey,
            # groups=[group_set[i] for i in range(len(bgc_ids))]
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
data.write_h5ad(args.output)
