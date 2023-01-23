import argparse
import collections
import io
import os
import csv
import gzip
import itertools
import tarfile
import json
import typing
import gzip
import multiprocessing.pool
import math
from pprint import pprint

import Bio.SeqIO
import pyrodigal
import rich.progress

parser = argparse.ArgumentParser()
parser.add_argument("--gbk", help="The GenBank file containing all records to annnotate", required=True)
parser.add_argument("-j", "--jobs", help="The number of threads to use to parallelize Pyrodigal", type=int, default=os.cpu_count() or 1)
parser.add_argument("-o", "--output", help="The name of the compressed FASTA file to generate", required=True)
args = parser.parse_args()

with rich.progress.Progress(
     rich.progress.SpinnerColumn(finished_text="[green]:heavy_check_mark:[/]"),
    "[progress.description]{task.description}",
    rich.progress.BarColumn(bar_width=60),
    rich.progress.MofNCompleteColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    rich.progress.TimeElapsedColumn(),
    rich.progress.TimeRemainingColumn(),
) as progress:

    # load genbank records
    with progress.open(args.gbk, "rb", description="Reading...") as f:
        if args.gbk.endswith(".gz"):
            f = gzip.open(f)
        f = io.TextIOWrapper(f)
        records = { 
            record.id: record 
            for i, record in enumerate(Bio.SeqIO.parse(f, "genbank")) 
        }

    # find ORFs
    orf_finder = pyrodigal.OrfFinder(meta=True)
    task = progress.add_task(total=len(records), description="Finding ORFs...")
    def find_genes(bgc_id, record):
        genes = orf_finder.find_genes(str(record.seq))
        progress.update(task_id=task, advance=1)
        return bgc_id, genes
    with multiprocessing.pool.ThreadPool(args.jobs) as pool:
        genes = dict(pool.starmap(find_genes, records.items()))

    # convert Pyrodigal genes to PyHMMER sequences
    with gzip.open(args.output, "wb") as dst:
        for bgc_id, bgc_genes in genes.items():
            for i, gene in enumerate(bgc_genes):
               name = f"{bgc_id}_{i+1}"
               prot = gene.translate().rstrip("*")
               seq = pyhmmer.easel.TextSequence(name=name.encode(), sequence=prot)
               seq.write(dst)
