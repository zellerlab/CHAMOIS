import argparse
import collections
import contextlib
import urllib.request
import os
import io
import gzip
import json
import posixpath
from itertools import islice

import joblib
import rich.progress
import rdkit.Chem
import pubchempy
import pandas
import Bio.SeqIO
from rdkit import RDLogger

# disable logging
RDLogger.DisableLog('rdApp.warning')

# get paths from command line
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
parser.add_argument("--clusters", required=True)
parser.add_argument("--atlas", required=True)
parser.add_argument("--cache")
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()

# create persistent cache
if args.cache:
    rich.print(f"[bold green]{'Using':>12}[/] joblib cache folder {args.cache!r}")
    os.makedirs(args.cache, exist_ok=True)
memory = joblib.Memory(location=args.cache, verbose=False)

# --- Load clusters with sequences -------------------------------------------

with contextlib.ExitStack() as ctx:

    progress = ctx.enter_context(rich.progress.Progress())
    output = ctx.enter_context(open(args.output, "w"))
    reader = ctx.enter_context(progress.open(args.clusters, "r", description=f"[bold blue]{'Reading':>12}[/]"))

    clusters = set()
    for record in Bio.SeqIO.parse(reader, "genbank"):
        clusters.add(record.id)


# --- Load compound structures -----------------------------------------------

rich.print(f"[bold blue]{'Loading':>12}[/] compounds from {args.input!r}")
data = pandas.read_excel(args.input, usecols=["Cluster", "True SMILES"]).drop_duplicates()
data["Cluster"] = data["Cluster"].str.replace("-", "_")#.str.split(".").str[0]
data = data[ data["Cluster"].isin(clusters) ]
rich.print(f"[bold green]{'Loaded':>12}[/] {data['Cluster'].nunique()} BGCs with known compounds")

# --- Compute Inchi and InchiKey ---------------------------------------------

inchi = []
inchikey = []

for row in rich.progress.track(data.itertuples(), total=len(data), description=f"[bold blue]{'Computing':>12}[/] InChi and InChi key"):
    _, cluster, smiles = row
    mol = rdkit.Chem.MolFromSmiles(smiles)
    inchi.append(rdkit.Chem.MolToInchi(mol))
    inchikey.append(rdkit.Chem.MolToInchiKey(mol))

data["Inchi"] = inchi
data["InchiKey"] = inchikey


# --- Load Natural Product Atlas ---------------------------------------------

with rich.progress.open(args.atlas, "rb", description=f"[bold blue]{'Loading':>12}[/] NPAtlas") as handle:
    np_atlas = { entry["inchikey"]: entry for entry in json.load(gzip.open(handle)) }


# --- Get metadata for each compound in NPAtlas or PubChem -------------------

# cache PubChem queries
@memory.cache
def get_cids(smiles):
    try:
        return pubchempy.get_cids(smiles, namespace="smiles")
    except pubchempy.BadRequestError:
        return []

@memory.cache
def get_compounds(cids):
    return pubchempy.get_compounds(cids)

compounds = collections.defaultdict(list)
for row in rich.progress.track(data.itertuples(), total=len(data), description=f"[bold blue]{'Working':>12}[/]"):
    _, cluster, smiles, inchi, inchikey = row
    name = cluster.replace("-", "_").split(".")[0]
    mol = rdkit.Chem.MolFromSmiles(smiles)

    # create compound for BGC
    compound = {"compound": name, "chem_struct": smiles}
    compounds[name].append(compound)

    # Search compound in NPAtlas
    entry = np_atlas.get(inchikey)
    if entry is not None:
        rich.print(f"[bold green]{'Mapped':>12}[/] {compound['compound']!r} to NPAtlas compound [bold cyan]{entry['npaid']}[/]")
        compound["database_id"] = [f"npatlas:{entry['npaid']}"]
        compound["compound"] = entry['original_name']
        continue

    # Search compound in PubChem
    with contextlib.suppress(pubchempy.BadRequestError):
        cids = get_cids(smiles)
        if cids:
            c = get_compounds(cids[:1])[0]
            compound["database_id"] = [f"pubchem:{cids[0]}"]
            synonyms = c.synonyms
            if synonyms:
                compound["compound"] = synonyms[0]
            rich.print(f"[bold green]{'Mapped':>12}[/] {compound['compound']!r} to PubChem compound [bold cyan]{c.cid}[/]")
            continue

    # failed to map
    rich.print(f"[bold red]{'Failed':>12}[/] to map {compound['compound']!r} to a database")


# --- Load Natural Product Atlas ---------------------------------------------

os.makedirs(os.path.dirname(args.output), exist_ok=True)

with open(args.output, "w") as dst:
    json.dump(compounds, dst, sort_keys=True, indent=4)
