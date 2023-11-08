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
import pubchempy
import rich.progress
import rapidfuzz.process
from rich.prompt import Confirm, Prompt
from rich.progress import TextColumn, BarColumn, MofNCompleteColumn, TaskProgressColumn, TimeRemainingColumn

parser = argparse.ArgumentParser()
parser.add_argument("--compounds", required=True)
parser.add_argument("--clusters", required=True)
parser.add_argument("--email", default="martin.larralde@embl.de")
args = parser.parse_args()

Bio.Entrez.email = args.email

rich.print(f"[bold blue]{'Loading':>12}[/] compounds from {str(args.compounds)}")
with open(args.compounds) as f:
    compounds = json.load(f)

# save records to output
rich.print(f"[bold blue]{'Downloading':>12}[/] {len(compounds)} BGC records")
os.makedirs(os.path.dirname(args.clusters), exist_ok=True)
accessions = { bgc_id.rsplit("_", 1)[0] for bgc_id in compounds.keys() }
with open(args.clusters, "w") as dst:
    with Bio.Entrez.efetch(db="nucleotide", id=",".join(accessions), rettype="gbwithparts", retmode="text") as handle:
        # reader = list(rich.progress.track(Bio.SeqIO.parse(handle, "genbank"), transient=True, total=len(accessions), description=f"[bold blue]{'Downloading':>12}[/]"))
        reader = Bio.SeqIO.parse(handle, "genbank")
        for record in rich.progress.track(reader, total=len(accessions), description=f"[bold blue]{'Downloading':>12}[/]"):
            # find feature for BGCs in the record, ignore multi-BGC record
            bgc_features = [
                (feat.qualifiers["note"][0], feat.location)
                for feat in record.features
                if feat.type == "misc_feature"
                and "biosynthetic gene cluster" in feat.qualifiers.get("note", [""])[0]
            ]
            if len(bgc_features) > 1:
                continue
            # save BGC feature if any
            if bgc_features:
                name, location = bgc_features[0]
                bgc_record = location.extract(record)
                bgc_record.annotations = record.annotations.copy()
                bgc_record.id = bgc_record.name = f"{record.id}_cluster1"
            else:
                bgc_record = record
                bgc_record.id = bgc_record.name = f"{record.id}_cluster1"
            Bio.SeqIO.write(bgc_record, dst, "genbank")




    
    
