import argparse
import contextlib
import statistics
import io
import tarfile
import os
import tempfile
import subprocess
import multiprocessing.pool

import anndata
import numpy
import pyfastani
import rich.progress
import scipy.sparse
import pandas
import Bio.SeqIO
import orthoani

parser = argparse.ArgumentParser()
parser.add_argument("-q", "--query", required=True)
parser.add_argument("-t", "--target", required=True)
parser.add_argument("-o", "--output", required=True)
parser.add_argument("-j", "--jobs", type=int)
args = parser.parse_args()

with rich.progress.Progress() as progress:

    # run many-to-many comparison
    with tempfile.TemporaryDirectory() as dst:

        # load query records
        with progress.open(args.query, "r", description=f"[bold blue]{'Loading':>12}[/]") as src:
            query_records = {
                record.id:record
                for record in Bio.SeqIO.parse(src, "genbank")
            }
            query_ids = sorted(query_records)
            query_indices = { name:i for i, name in enumerate(query_ids) }

        # load target records
        with progress.open(args.target, "r", description=f"[bold blue]{'Loading':>12}[/]") as src:
            target_records = {
                record.id:record
                for record in Bio.SeqIO.parse(src, "genbank")
            }
            target_ids = sorted(target_records)
            target_indices = { name:i for i, name in enumerate(target_ids) }

        # write target sequences to FASTA
        db_filename = os.path.join(dst, "db.fna")
        Bio.SeqIO.write(target_records.values(), db_filename, "fasta")
        # make blastd
        proc = subprocess.run(["makeblastdb", "-in", db_filename, "-dbtype", "nucl"], capture_output=True)
        proc.check_returncode()
       
        # load query records
        with multiprocessing.pool.ThreadPool(args.jobs) as pool:
            def process(query_record):
                # write query sequences to FASTA
                with tempfile.NamedTemporaryFile(suffix=".fna") as dst:
                    Bio.SeqIO.write(query_record, dst.name, "fasta")
                    # run BLASTn
                    proc = subprocess.run([
                        "blastn",
                        "-task",
                        "megablast",
                        "-query",
                        dst.name,
                        "-db",
                        db_filename,
                        "-perc_identity",
                        "50",
                        "-subject_besthit",
                        "-qcov_hsp_perc",
                        "50",
                        "-outfmt",
                        "7"
                    ], capture_output=True)        
                    proc.check_returncode()
                    # read results
                    return pandas.read_table(
                        io.BytesIO(proc.stdout), 
                        comment="#", 
                        header=None,
                        index_col=None,
                        names=[
                            "query", 
                            "subject",
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
            hits = pandas.concat(progress.track(pool.imap(process, query_records.values()), total=len(query_records), description=f"[bold blue]{'Matching':>12}[/]"))

    # patch query and subject accessions
    hits["query"] = hits["query"].str.rsplit(".", 1).str[0]
    hits["subject"] = hits["subject"].str.rsplit(".", 1).str[0]

    # create matrix identity
    identity = scipy.sparse.dok_matrix((len(query_records), len(target_records)), dtype=numpy.float_)

    # compute ANI
    for row in hits.itertuples():
        i = query_indices[row.query]
        j = target_indices[row.subject]
        identity[i, j] = row.identity / 100.0

    # generate annotated data
    data = anndata.AnnData(
        dtype=numpy.float_,
        X=identity.tocsr(),
        obs=pandas.DataFrame(index=query_ids),
        var=pandas.DataFrame(index=target_ids),
    )

    # save annotated data
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    data.write(args.output)
