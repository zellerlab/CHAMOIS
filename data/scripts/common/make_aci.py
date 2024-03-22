import argparse
import contextlib
import statistics
import io
import tarfile
import os
import sys
import tempfile
import subprocess
import multiprocessing.pool

import anndata
import iocursor
import gb_io
import numpy
import rich.panel
import rich.progress
import scipy.sparse
import pandas
import pyrodigal
import Bio.SeqIO

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..", "..")))
import chamois.orf
from chamois.model import ClusterSequence
from chamois.cli._common import find_proteins

parser = argparse.ArgumentParser()
parser.add_argument("-q", "--query", required=True)
parser.add_argument("-t", "--target", required=True)
parser.add_argument("-o", "--output", required=True)
parser.add_argument("-j", "--jobs", type=int)
args = parser.parse_args()

# run many-to-many comparison
with tempfile.TemporaryDirectory() as dst:
    
    with rich.progress.Progress() as progress:
        # load query records
        with progress.open(args.query, "r", description=f"[bold blue]{'Loading':>12}[/]") as src:
            query_records = {
                record.name:ClusterSequence(record)
                for record in gb_io.iter(src)
            }
            query_ids = sorted(query_records)
            query_indices = { name:i for i, name in enumerate(query_ids) }
        # load target records
        with progress.open(args.target, "r", description=f"[bold blue]{'Loading':>12}[/]") as src:
            target_records = {
                record.name:ClusterSequence(record)
                for record in gb_io.iter(src)
            }
            target_ids = sorted(target_records)
            target_indices = { name:i for i, name in enumerate(target_ids) }

    # find genes
    progress.console.print(f"[bold blue]{'Finding':>12}[/] genes in input records")
    orf_finder = chamois.orf.PyrodigalFinder(cpus=args.jobs)
    query_genes = find_proteins(list(query_records.values()), orf_finder, progress.console)
    target_genes = find_proteins(list(target_records.values()), orf_finder, progress.console)

    # record protein length
    protein_lengths = {
        protein.id:len(protein.sequence)
        for proteins in (query_genes, target_genes)
        for protein in proteins
    }

    # write target genes
    progress.console.print(f"[bold blue]{'Writing':>12}[/] protein sequences")
    db_faa = os.path.join(dst, "db.faa")
    with open(db_faa, "w") as f:
        for protein in progress.track(target_genes, description=f"[bold blue]{'Writing':>12}[/]"):
            f.write(f">{protein.id}\n")                
            f.write(f"{protein.sequence.rstrip('*')}\n")                
    
    # make blastd
    db_filename = os.path.join(dst, "db.db")
    proc = subprocess.run([
        "diamond", 
        "makedb", 
        "--in", 
        db_faa, 
        "--db",
        db_filename,
        "--tmpdir",
        dst,
        "--threads", 
        str(args.jobs or os.cpu_count())
    ], capture_output=True)
    if proc.returncode != 0:
        rich.print(rich.panel.Panel(proc.stderr.decode()))
    proc.check_returncode()

    # write query records
    query_filename = os.path.join(dst, "query.faa")
    with open(query_filename, "w") as f:
        for protein in progress.track(query_genes, description=f"[bold blue]{'Writing':>12}[/]"):
            f.write(f">{protein.id}\n")                
            f.write(f"{protein.sequence.rstrip('*')}\n")                

    # run BLASTn
    proc = subprocess.run([
        "diamond",
        "blastp",
        "--query",
        query_filename,
        "--db",
        db_filename,
        "--threads", 
        str(args.jobs or os.cpu_count()),
        "--outfmt",
        "6",
        "--max-target-seqs",
        str(len(target_ids)),
        "--tmpdir",
        dst,
    ], capture_output=True)
    if proc.returncode != 0:
        rich.print(rich.panel.Panel(proc.stderr.decode()))
    proc.check_returncode()

    hits = pandas.read_table(
        iocursor.Cursor(proc.stdout),
        comment="#",
        header=None,
        index_col=None,
        names=[
            "query_protein",
            "target_protein",
            "identity",
            "alilen",
            "mismatches",
            "gapopens",
            "qstart",
            "qend",
            "sstart",
            "send",
            "evalue",
            "bitscore"
        ]
    )
    hits["query_cluster"] = hits["query_protein"].str.rsplit("_", n=1).str[0]
    hits["target_cluster"] = hits["target_protein"].str.rsplit("_", n=1).str[0]
    hits["query_index"] = hits["query_cluster"].map(query_indices.__getitem__)
    hits["target_index"] = hits["target_cluster"].map(target_indices.__getitem__)

# only keep one hit per query protein per target cluster
hits = (
    hits
        .sort_values(["query_protein", "bitscore"])
        .drop_duplicates(["query_protein", "target_cluster"], keep="last")
)

# compute identity
identity = numpy.zeros((len(query_records), len(target_records)), dtype=numpy.float_)
for row in hits.itertuples():
    identity[row.query_index, row.target_index] += row.identity * row.alilen / 100.0

# normalize by query length
for i, query_id in enumerate(query_ids):
    query_length = sum( 
        len(protein.sequence) 
        for protein in query_genes 
        if protein.cluster.id == query_id
    )
    if query_length > 0:
        identity[i] /= query_length

# make distances symmetric
identity = numpy.clip(identity, 0, 1)

# generate annotated data
data = anndata.AnnData(
    dtype=numpy.float_,
    X=scipy.sparse.csr_matrix(identity),
    obs=pandas.DataFrame(index=query_ids),
    var=pandas.DataFrame(index=target_ids),
)

# save annotated data
os.makedirs(os.path.dirname(args.output), exist_ok=True)
data.write(args.output)
