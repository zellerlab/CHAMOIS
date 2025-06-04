import argparse
import itertools
import json
import gzip
import csv
import pickle
import os
import time
import math
import random
import tarfile
import urllib.request

import Bio.Entrez
import Bio.SeqIO
import pandas
import rich.progress
from rich.prompt import Confirm, Prompt
from rich.progress import TextColumn, BarColumn, MofNCompleteColumn, TaskProgressColumn, TimeRemainingColumn

parser = argparse.ArgumentParser()
parser.add_argument("--compounds", required=True)
parser.add_argument("--clusters", required=True)
parser.add_argument("--coordinates", required=True)
parser.add_argument("--cache", required=False)
parser.add_argument("--email", default="martin.larralde@embl.de")
args = parser.parse_args()

Bio.Entrez.email = args.email

rich.print(f"[bold blue]{'Loading':>12}[/] compounds from {str(args.compounds)}")
with open(args.compounds) as f:
    compounds = json.load(f)

# load cluster coordinates
coordinates = pandas.read_table(args.coordinates)

# save records to output
rich.print(f"[bold blue]{'Downloading':>12}[/] {len(compounds)} BGC records")
os.makedirs(os.path.dirname(args.clusters), exist_ok=True)
accessions = { bgc_id.rsplit("_", 1)[0] for bgc_id in compounds.keys() }
with open(args.clusters, "w") as dst:
    for row in rich.progress.track(coordinates.itertuples(), total=len(coordinates), description=f"[bold blue]{'Downloading':>12}[/]"):
        with Bio.Entrez.efetch(db="nucleotide", id=row.sequence_id, seq_start=row.start, seq_stop=row.end, rettype="gbwithparts", retmode="text") as handle:
            bgc_record = Bio.SeqIO.read(handle, "genbank")
            bgc_record.id = bgc_record.name = row.bgc_id
            Bio.SeqIO.write(bgc_record, dst, "genbank")
        
