import argparse
import itertools
import json
import gzip
import csv
import pickle
import os
import re
import time
import math
import random
import tarfile
import urllib.error
import shutil

import Bio.Entrez
import Bio.SeqIO
import pandas
import rich.progress
from rich.prompt import Confirm, Prompt
from rich.progress import TextColumn, BarColumn, MofNCompleteColumn, TaskProgressColumn, TimeRemainingColumn

parser = argparse.ArgumentParser()
#parser.add_argument("--compounds", required=True)
parser.add_argument("--genomes", required=True)
parser.add_argument("--coordinates", required=True)
parser.add_argument("--email", default="martin.larralde@embl.de")
args = parser.parse_args()

Bio.Entrez.email = args.email

rich.print(f"[bold blue]{'Loading':>12}[/] clusters from {str(args.coordinates)}")
coordinates = pandas.read_table(args.coordinates)
genomes = coordinates["genome"].unique()

# save records to output
rich.print(f"[bold blue]{'Downloading':>12}[/] {len(genomes)} BGC records")
os.makedirs(os.path.dirname(args.genomes), exist_ok=True)

for genome in rich.progress.track(genomes, description=f"[bold blue]{'Downloading':>12}[/]"):
    genome_path = os.path.join(args.genomes, f"{genome}.fna")
    if not os.path.exists(genome_path):
        if len(genome) == 10:
            with Bio.Entrez.efetch(db="nucleotide", id=genome, rettype="fasta", retmode="text") as src:
                with open(genome_path, "w") as dst:
                    shutil.copyfileobj(src, dst)
                    continue

        else:

             base_url = f"https://sra-download.ncbi.nlm.nih.gov/traces"
             path = []
             for i in range(0, len(genome), 2):
                 if genome[i:i+2].isdigit():
                     break
                 path.append(genome[i:i+2])

             for i in range(1, 5):
                 url = f"{base_url}wgs{i:02}/wgs_aux/{'/'.join(path)}/{genome}/{genome}.1.fsa_nt.gz"
                 rich.print(f"[bold blue]{'Trying':>12}[/] URL {url!r}")
                 try:
                     with urllib.request.urlopen(url) as res:
                         with gzip.open(res, "rt") as src:
                             with open(genome_path, "w") as dst:
                                 shutil.copyfileobj(src, dst)
                                 break
                 except urllib.request.HTTPError:
                     pass
             if not os.path.exists(genome_path):
                 rich.print(f"[bold red]{'Failed':>12}[/] to download {genome!r}")

