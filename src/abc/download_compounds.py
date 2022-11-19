import argparse
import os
import io
import itertools
import json
import shutil
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import multiprocessing.pool

import Bio.Entrez
import Bio.SeqIO
import requests
import rich.console
import rich.progress
from bs4 import BeautifulSoup

CGI_URL = "https://img.jgi.doe.gov/cgi-bin"

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()

# Create a session and spoof the user agent so that JGI lets us use programmatic access
console = rich.console.Console()
session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64; rv:78.0) Gecko/20100101 Firefox/78.0"

# Load input clusters
console.print(f"[bold green]{'Loading':>12}[/] IMG/ABC cluster from {args.input!r}")
with open(args.input) as f:
    imgabc = json.load(f)

console.print(f"[bold green]{'Downloading':>12}[/] IMG/ABC compounds")
with rich.progress.Progress(
     rich.progress.SpinnerColumn(finished_text="[green]:heavy_check_mark:[/]"),
     "[progress.description]{task.description}",
     rich.progress.BarColumn(bar_width=60),
     "[progress.completed]{task.completed}/{task.total}",
     "[progress.percentage]{task.percentage:>3.0f}%",
     rich.progress.TimeElapsedColumn(),
     rich.progress.TimeRemainingColumn(),
     console=console,
     transient=True,
) as progress:

    oids = {}
    for img_bgc in progress.track(imgabc, total=len(imgabc), description=f"{'Loading':>12}"):
        # find compound OIDs in BGC
        compound_section = BeautifulSoup(img_bgc["SecondaryMetaboliteDisp"], "html.parser")
        oids[img_bgc['ClusterID']] = [
            urllib.parse.parse_qs(urllib.parse.urlparse(link.attrs["href"]).query)["compound_oid"][0]
            for link in compound_section.find_all("a")
        ]

    unique_oids = set(itertools.chain.from_iterable(oids.values()))
    with multiprocessing.pool.ThreadPool() as pool:
        task = progress.add_task(total=len(unique_oids), description=f"{'Downloading':>12}")

        def download_compound(oid: str):
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
            compound = {
                "compound": compound_data["Compound Name"],
            }
            if compound_data.get("SMILES"):
                compound["chem_struct"] = compound_data["SMILES"]
            if compound_data.get("PubChem Compound"):
                compound.setdefault("database_id", []).append(f"pubchem:{compound_data['PubChem Compound']}")
            if compound_data.get("ChEBI"):
                compound.setdefault("database_id", []).append(f"chebi:{compound_data['ChEBI']}")
            # update progress bar
            progress.console.print(f"{'Downloaded':>12} metadata for IMG compound {oid} ({compound['compound']})")
            progress.update(task_id=task, advance=1)
            return oid, compound

        oid_data = dict(pool.map(download_compound, unique_oids))
        progress.update(task_id=task, visible=False)

    compounds = {
        bgc_id: [ oid_data[oid] for oid in bgc_oids ]
        for bgc_id, bgc_oids in oids.items()
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as dst:
        json.dump(compounds, dst, indent=4, sort_keys=True)
        

