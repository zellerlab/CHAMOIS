import argparse
import os
import io
import json
import shutil
import tarfile
import urllib.error
import urllib.request
import multiprocessing.pool
import time

import Bio.Entrez
import Bio.SeqIO
import requests
import rich.console
import rich.progress
from bs4 import BeautifulSoup

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()

# Create a session and spoof the user agent so that JGI lets us use programmatic access
console = rich.console.Console()
session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64; rv:78.0) Gecko/20100101 Firefox/78.0"

# Setup Entrez email to avoid rate limits on queries
Bio.Entrez.email = "martin.larralde@embl.de"

# Load input clusters
console.print(f"[bold green]{'Loading':>12}[/] IMG/ABC cluster from {args.input!r}")
with open(args.input) as f:
    imgabc = json.load(f)

console.print(f"[bold green]{'Downloading':>12}[/] BGCs to [purple]GenBank[/] format")
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

    # extract list of GenBank accessions to download
    accessions = set()
    for img_bgc in imgabc:
        accessions.add(img_bgc["GenbankID"])

    # batch download from GenBank
    records = {}
    with Bio.Entrez.efetch(db="nucleotide", id=",".join(accessions), rettype="gbwithparts", retmode="text") as handle:
        reader = progress.track(Bio.SeqIO.parse(handle, "genbank"), total=len(accessions), description=f"[bold green]{'Downloading':>12}[/]")
        for record in reader:
            accession, version = record.id.rsplit(".", 1)
            records[accession] = record
    progress.console.print(f"[bold green]{'Downloaded':>12}[/] {len(records)} GenBank records (expected {len(accessions)})")

    # collect records
    bgc_records = []
    for img_bgc in progress.track(imgabc, description=f"[bold blue]{'Extracting':>12}[/]"):
        # get record 
        record = records.get(img_bgc["GenbankID"], None)
        start = int(img_bgc["StartCoord"])
        end = int(img_bgc["EndCoord"])
        if record is None or not record[start:end].seq:
            # try to download the record from the ABC page if downloading it from GenBank failed
            params = dict(section="BiosyntheticDetail", page="geneExport", taxon_oid=img_bgc['GenomeID'], cluster_id=img_bgc['ClusterID'], fasta="genbank")
            with session.get("https://img.jgi.doe.gov/cgi-bin/abc-public/main.cgi", params=params) as res:
                soup = BeautifulSoup(res.text, "html.parser")
                content = soup.find("pre").text
                bgc_record = Bio.SeqIO.read(io.StringIO(content), "genbank")
                assert len(bgc_record.seq) > 0
            progress.console.print(f"[bold green]{'Recovered':>12}[/] GenBank record of [purple]{img_bgc['ClusterID']}[/] from IMG-ABC")
        else:
            # extract region of interest from the GenBank record
            bgc_record = record[start:end]
            progress.console.print(f"[bold blue]{'Extracting':>12}[/] BGC at coordinates {start} to {end}")
            # copy annotations, since Biopython doesn't do it by default
            bgc_record.annotations = record.annotations.copy()
        # check the record is not empty
        if not bgc_record.seq:
            progress.console.print(f"[bold red]{'Failed':>12}[/] getting record sequence for {img_bgc['ClusterID']}")
            exit(1)
        # set the accession and identifier of the BGC
        bgc_record.id = bgc_record.name = img_bgc['ClusterID']
        # advance progress bar once finished
        bgc_records.append(bgc_record)
        
    # save it to output file
    console.print(f"[bold blue]{'Saving':>12}[/] GenBank records to {args.output!r}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as dst:
        n = Bio.SeqIO.write(bgc_records, dst, "genbank")
    console.print(f"[bold green]{'Saved':>12}[/] {n} GenBank records")

