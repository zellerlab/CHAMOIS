import argparse
import urllib.error
import urllib.request
import json
import pathlib
import itertools
import io
import os
import sys

import Bio.Entrez
import Bio.SeqIO
import gb_io
import rich.progress
from rich.console import Console

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..", "..")))

from chamois.model import ClusterSequence, Protein
from chamois.compositions import build_observations, build_variables, build_compositions
from chamois.cli.annotate import save_compositions
from chamois.cli._common import (
    annotate_hmmer,
    find_proteins,
    load_sequences,
    record_metadata,
)

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", required=True, type=pathlib.Path)
parser.add_argument("--output", "-o", required=True, type=pathlib.Path)
parser.add_argument("--hmm", required=True, type=pathlib.Path)
parser.add_argument("--jobs", "-j", type=int, default=0)
args = parser.parse_args()

console = Console()

with args.input.open() as src:
    peptides = json.load(src)

proteins = []
clusters = []

for entry in peptides:
    cluster = ClusterSequence(
        record=gb_io.Record(b"N" * 1000, name=entry["accession"])
    )
    clusters.append(cluster)
    for enzyme in entry["enzymes"]:
        id_ = enzyme["ids"].get("uniprot") or enzyme["ids"].get("genpept")
        prot = Protein(id_, enzyme["sequence"], cluster=cluster)
        proteins.append(prot)

uns = record_metadata()    
domains = annotate_hmmer(args.hmm, proteins, args.jobs, console)
obs = build_observations(clusters, proteins)
var = build_variables(domains)
compositions = build_compositions(domains, obs, var, uns=uns)
save_compositions(compositions, args.output, console)