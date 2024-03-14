import argparse
import contextlib
import urllib.request
import os
import io
import tarfile
import posixpath

import pandas
import rich.progress
import Bio.SeqIO


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
parser.add_argument("--table", required=True)
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()

# --- Load compound structures -----------------------------------------------

rich.print(f"[bold blue]{'Loading':>12}[/] compounds from {args.input!r}")
data = pandas.read_excel(args.table, usecols=["Cluster", "True SMILES"])
cluster_files = set(data["Cluster"].unique())
rich.print(f"[bold green]{'Loaded':>12}[/] {data['Cluster'].nunique()} BGCs with known compounds")


# --- Load clusters with sequences -------------------------------------------

with contextlib.ExitStack() as ctx:

    progress = ctx.enter_context(rich.progress.Progress())
    reader = ctx.enter_context(progress.open(args.input, "rb"))
    tar = ctx.enter_context(tarfile.open(fileobj=reader, mode="r"))
    output = ctx.enter_context(open(args.output, "w"))
    
    for entry in tar:
        basename = posixpath.basename(entry.name)
        if basename in cluster_files:
            fmt = "genbank" if basename.endswith(".gbk") else "fasta"
            name, _ = posixpath.splitext(basename)
            with tar.extractfile(entry) as f:
                record = next(Bio.SeqIO.parse(io.TextIOWrapper(f), fmt))
                record.id = record.name = name               
                record.annotations["molecule_type"] = "DNA"     
          
            Bio.SeqIO.write(record, output, "genbank")


