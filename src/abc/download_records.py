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
    console.print(f"[bold green]{'Downloaded':>12}[/] {len(records)} GenBank records (expected {len(accessions)})")

    # 
    bgc_records = []
    for img_bgc in imgabc:
        # get record 
        record = records.get(img_bgc["GenbankID"], None)
        if record is None:
            # try to download the record from the ABC page if downloading it from GenBank failed
            params = dict(section="BiosyntheticDetail", page="geneExport", taxon_oid=img_bgc['GenomeID'], cluster_id=img_bgc['ClusterID'], fasta="genbank")
            with session.get("https://img.jgi.doe.gov/cgi-bin/abc-public/main.cgi", params=params) as res:
                soup = BeautifulSoup(res.text, "html.parser")           
                record = Bio.SeqIO.read(io.StringIO(soup.find("pre").text), "genbank")
            progress.console.print(f"[bold green]{'Recovered':>12}[/] GenBank record of [purple]{img_bgc['ClusterID']}[/] from IMG-ABC")

        # Extract only the BGC from the record, as sometimes the record contains 
        # unrelated flanking genes. Note that to do so we use the internal 
        # BGC annotations inside the GenBank record if there are any instead of 
        # the IMG-ABC annotations, because the latter are less trustworthy by
        # experience.
        for feature in record.features:
            if feature.type != "misc_feature":
                continue
            if feature.qualifiers.get("note", [""])[0].endswith("gene cluster"):
                bgc_record = feature.location.extract(record)
                break
        else:
            start = int(img_bgc["StartCoord"])
            end = int(img_bgc["EndCoord"])
            bgc_record = record[start:end]

        # copy annotations, since Biopython doesn't do it by default
        bgc_record.annotations = record.annotations.copy()

        # set the accession and identifier of the BGC
        record.id = img_bgc['ClusterID']
        if img_bgc["GenbankID"]:
            record.accession = img_bgc["GenbankID"]

        # advance progress bar once finished
        bgc_records.append(record)
        

    # save it to output file
    console.print(f"[bold green]{'Saving':>12}[/] GenBank records to {args.output!r}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as dst:
        Bio.SeqIO.write(bgc_records, dst, "genbank")
        

