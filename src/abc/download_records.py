import argparse
import os
import io
import json
import shutil
import tarfile
import urllib.error
import urllib.request

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

console.print(f"[bold green]{'Downloading':>12}[/] IMG/ABC BGCs to [purple]GenBank[/] format")
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

    with open(args.output, "w") as dst:

        for img_bgc in progress.track(imgabc, total=len(imgabc), description="Downloading..."):       
            # Download the BGC from GenBank
            progress.console.print(f"[bold green]{'Downloading':>12}[/] cluster [purple]{img_bgc['ClusterID']}[/]")
            try:
                with Bio.Entrez.efetch(db="nucleotide", id=img_bgc["GenbankID"], rettype="gbwithparts", retmode="text") as handle:
                    record = Bio.SeqIO.read(handle, "genbank")
            except urllib.error.HTTPError:
                progress.console.print(f"[bold red]{'Failed':>12}[/] downloading GenBank entry [purple]{img_bgc['GenbankID']}[/]")
                continue
            except ValueError:
                progress.console.print(f"[bold red]{'Failed':>12}[/] parsing GenBank entry [purple]{img_bgc['GenbankID']}[/]")
                continue
                
            # extract only the BGC from the record, as sometimes the 
            # record contains unrelated flanking
            bgc_feature = next(
                (
                    feat for feat in record.features
                    if feat.type == "misc_feature"
                    and feat.qualifiers.get("note", [""])[0].endswith("gene cluster")
                ),
                None
            )
            if bgc_feature is not None:
                bgc_record = bgc_feature.location.extract(record)
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

            # save it to output file
            Bio.SeqIO.write(record, dst, "genbank")
        

