import argparse
import tarfile
import collections
import json
import gzip
import urllib.request
import time
import os

import joblib
import rich.progress
import pubchempy
import pandas

# get paths from command line
parser = argparse.ArgumentParser()
parser.add_argument("--mibig-version", default="4.0", choices={"1.3", "2.0", "3.1", "4.0"})
parser.add_argument("--blocklist")
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()

# --- Download MIBiG metadata ------------------------------------------------

url = f"https://dl.secondarymetabolites.org/mibig/mibig_json_{args.mibig_version}.tar.gz"

with rich.progress.Progress() as progress:
    # load blocklist if any
    if args.blocklist is not None:
        table = pandas.read_table(args.blocklist)
        blocklist = set(table.bgc_id.unique())
    else:
        blocklist = set()

    # download MIBIG 3 records
    mibig = {}
    with urllib.request.urlopen(url) as response:
        total = int(response.headers["Content-Length"])
        with progress.wrap_file(response, total=total, description=f"[bold blue]{'Downloading':>12}[/] MIBiG") as f:
            with tarfile.open(fileobj=f, mode="r|gz") as tar:
                for entry in iter(tar.next, None):
                    if entry.name.endswith(".json"):
                        with tar.extractfile(entry) as f:
                            record = json.load(f)
                            if "cluster" in record:
                                record = record["cluster"]
                            accession_key = "mibig_accession" if "mibig_accession" in record else "accession"
                            if record[accession_key] not in blocklist:
                                mibig[record[accession_key]] = record
                            if "biosynthesis" in record:
                                record["biosyn_class"] = [ 

                                    "RiPP" if c["class"] == "ribosomal" else 
                                    "Terpene" if c["class"] == "terpene" else
                                    "Polyketide" if c["class"] == "PKS" else
                                    "NRP" if c["class"] == "NRPS" else
                                    "Other" if c["class"] == "other" else
                                    "Saccharide" if c["class"] == "saccharide" else
                                    c["class"]
                                    for c in record["biosynthesis"]["classes"] 
                                ]

    # write results
    with open(args.output, "w") as dst:
        for record in mibig.values():
            print(record[accession_key], ";".join(sorted(record["biosyn_class"])), file=dst, sep="\t")