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
rich.print(f"[bold green]{'Loaded':>12}[/] {len(cluster_files)} BGCs with known compounds")


# --- Load clusters with sequences -------------------------------------------

with contextlib.ExitStack() as ctx:

    progress = ctx.enter_context(rich.progress.Progress())
    reader = ctx.enter_context(progress.open(args.input, "rb", description=f"[bold blue]{'Reading':>12}[/]"))
    tar = ctx.enter_context(tarfile.open(fileobj=reader, mode="r"))
    output = ctx.enter_context(open(args.output, "w"))
    done = set()

    for entry in tar:
        basename = posixpath.basename(entry.name)
        if basename in cluster_files:
            fmt = "genbank" if basename.endswith(".gbk") else "fasta"
            name = basename.replace("-", "_").split(".")[0]
            with tar.extractfile(entry) as f:
                record = next(Bio.SeqIO.parse(io.TextIOWrapper(f), fmt))
                record.id = record.name = name               
                record.annotations["molecule_type"] = "DNA"     
          
            if name not in done:
                Bio.SeqIO.write(record, output, "genbank")
                done.add(record.id)

    rich.print(f"[bold green]{'Extracted':>12}[/] {len(done)} BGC records")
    