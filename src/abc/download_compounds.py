import argparse
import os
import io
import itertools
import gzip
import json
import shutil
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import multiprocessing.pool
import functools

import Bio.Entrez
import Bio.SeqIO
import joblib
import requests
import rich.console
import rich.progress
import rdkit.Chem
from bs4 import BeautifulSoup
from rdkit import RDLogger

# disable logging
RDLogger.DisableLog('rdApp.warning')  

CGI_URL = "https://img.jgi.doe.gov/cgi-bin"

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
parser.add_argument("-o", "--output", required=True)
parser.add_argument("--atlas", required=True)
parser.add_argument("--cache")
args = parser.parse_args()

# Create a session and spoof the user agent so that JGI lets us use programmatic access
console = rich.console.Console()
session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64; rv:78.0) Gecko/20100101 Firefox/78.0"

# create persistent cache
if args.cache:
    console.print(f"[bold green]{'Using':>12}[/] joblib cache folder {args.cache!r}")
    os.makedirs(args.cache, exist_ok=True)
memory = joblib.Memory(location=args.cache, verbose=False)

# --- Load NPAtlas -----------------------------------------------------------

with rich.progress.open(args.atlas, "rb", description=f"[bold blue]{'Loading':>12}[/] NPAtlas") as handle:
    data = json.load(gzip.open(handle))
    np_atlas = { entry["npaid"]: entry for entry in data }
    np_atlas_inchikeys = { entry["inchikey"]: entry for entry in data }

# --- Load input clusters ----------------------------------------------------

console.print(f"[bold green]{'Loading':>12}[/] IMG/ABC cluster from {args.input!r}")
with open(args.input) as f:
    imgabc = json.load(f)


# --- Download JGI compounds -------------------------------------------------

# download a single compound from the JGI, with caching to avoid downloading
# compounds that we already downloaded
@memory.cache
def download_compound(oid):
    # load compound metadata from the JGI website
    params = dict(section='ImgCompound', page='imgCpdDetail', compound_oid=oid)
    with session.get(f"{CGI_URL}/abc-public/main.cgi", params=params) as res:
        soup = BeautifulSoup(res.text, "html.parser")
    # parse the table and extract key/value pairs
    table = soup.find("table", class_="img")
    compound_data = {}
    for row in table.find_all("tr"):
        key = row.find("th").text.strip()
        value = row.find("td").text.strip()
        compound_data[key] = value
    # convert into a format similar to MIBiG compounds
    compound = { "compound": compound_data["Compound Name"], "database_id": [] }
    if compound_data.get("SMILES"):
        compound["chem_struct"] = compound_data["SMILES"]
    if compound_data.get("PubChem Compound"):
        compound["database_id"].append(f"pubchem:{compound_data['PubChem Compound']}")
    if compound_data.get("ChEBI"):
        compound["database_id"].append(f"chebi:{compound_data['ChEBI']}")
    return oid, compound

# download compound data in parallel
console.print(f"[bold green]{'Downloading':>12}[/] IMG/ABC compounds")
with rich.progress.Progress(
     "[progress.description]{task.description}",
     rich.progress.BarColumn(bar_width=60),
     "[progress.completed]{task.completed}/{task.total}",
     "[progress.percentage]{task.percentage:>3.0f}%",
     rich.progress.TimeElapsedColumn(),
     rich.progress.TimeRemainingColumn(),
     console=console,
     transient=True,
) as progress:
    # extract all compound IDs
    oids = {}
    for img_bgc in progress.track(imgabc, total=len(imgabc), description=f"[bold blue]{'Loading':>12}[/]"):
        # find compound OIDs in BGC
        compound_section = BeautifulSoup(img_bgc["SecondaryMetaboliteDisp"], "html.parser")
        oids[img_bgc['ClusterID']] = [
            urllib.parse.parse_qs(urllib.parse.urlparse(link.attrs["href"]).query)["compound_oid"][0]
            for link in compound_section.find_all("a")
        ]
    # recover data for all compound IDs
    unique_oids = sorted(set(itertools.chain.from_iterable(oids.values())))
    with multiprocessing.pool.ThreadPool() as pool:
        task = progress.add_task(total=len(unique_oids), description=f"[bold blue]{'Downloading':>12}[/]")
        oid_data = dict(progress.track(
            pool.imap(download_compound, unique_oids),
            task_id=task,
        ))
        progress.update(task_id=task, visible=False)
    # patch with NPAtlas cross-references
    for oid, compound in oid_data.items():
        if "chem_struct" in compound:
            inchikey = rdkit.Chem.inchi.MolToInchiKey(rdkit.Chem.MolFromSmiles(compound['chem_struct'].strip()))
            if inchikey in np_atlas_inchikeys:
                npaid = np_atlas_inchikeys[inchikey]["npaid"]
                compound["database_id"].append(f"npatlas:{npaid}")


# --- Save results -----------------------------------------------------------

compounds = {
    bgc_id: [ oid_data[oid] for oid in bgc_oids ]
    for bgc_id, bgc_oids in oids.items()
}

rich.print(f"[bold green]{'Saving':>12}[/] results to {args.output!r}")
os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as dst:
    json.dump(compounds, dst, indent=4, sort_keys=True)


