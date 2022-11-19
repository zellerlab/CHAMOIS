import argparse
import collections
import io
import os
import csv
import itertools
import tarfile
import json
import typing
import gzip
import multiprocessing.pool
import math
from pprint import pprint

import anndata
import Bio.SeqIO
import numpy
import scipy.sparse
import pandas
import pyhmmer
import pyrodigal
import rich.progress
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

parser = argparse.ArgumentParser()
parser.add_argument("--hmm", help="The HMM to use for annotating proteins", required=True)
parser.add_argument("--gbk", help="The GenBank file containing all records to annnotate", required=True)
# parser.add_argument("--json", required=True)
parser.add_argument("-j", "--jobs", help="The number of threads to use to parallelize Pyrodigal and PyHMMER", type=int, default=os.cpu_count() or 1)
parser.add_argument("--p-value", help="The p-value threshold for retaining domains", type=float, default=1e-9)
parser.add_argument("-o", "--output", help="The name of the file to generate", required=True)
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
    with progress.open(args.gbk, description="Reading...") as f:
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
    sequences = {}
    alphabet = pyhmmer.easel.Alphabet.amino()
    for bgc_id, bgc_genes in genes.items():
        sequences[bgc_id] = [
            pyhmmer.easel.TextSequence( 
                name=f"{bgc_id}_{i+1}".encode(),
                sequence=gene.translate().rstrip("*")
            ).digitize(alphabet)
            for i, gene in enumerate(bgc_genes)
        ]

    # extract all possible HMM names
    hmm_names = {}

    # annotate genes with Pfam domains
    all_sequences = list(itertools.chain.from_iterable(sequences.values()))
    domain_counts = collections.defaultdict(collections.Counter)
    task = progress.add_task(total=None, description="Finding domains...")
    with pyhmmer.plan7.HMMFile(args.hmm) as hmm_file:
        def callback(hmm, total):
            progress.update(task_id=task, advance=1, total=total)
        for hits in pyhmmer.hmmer.hmmsearch(hmm_file, all_sequences, callback=callback, cpus=args.jobs):
            hmm_accession = hits.query_accession.decode()
            hmm_name = hits.query_name.decode()
            hmm_names[hmm_accession] = hmm_name
            for hit in hits:
                if hit.pvalue < args.p_value:
                    sequence_id = hit.name.decode()
                    bgc_id = sequence_id.split("_", 1)[0]
                    domain_counts[bgc_id][hmm_accession] += 1
            break
            
    # record all HMMs from the file
    all_possible = sorted(hmm_names)
    name_index = { name:i for i,name in enumerate(all_possible) }

    # compute domain counts
    bgc_ids = sorted(records)
    counts = scipy.sparse.dok_matrix((len(bgc_ids), len(all_possible)), dtype=numpy.int32)
    for i, bgc_id in enumerate(bgc_ids):
        for domain_name, domain_count in domain_counts[bgc_id].items():
            j = name_index[domain_name]
            counts[i, j] = domain_count

    # generate annotated data
    data = anndata.AnnData(
        dtype=numpy.int32,
        X=counts.tocsr(),
        obs=pandas.DataFrame(
            index=bgc_ids,
            data=dict(
                source=[records[bgc_id].annotations["source"] for bgc_id in bgc_ids],
            )
        ),
        var=pandas.DataFrame(
            index=numpy.array(all_possible),
            data=dict(
                name=[hmm_names[accession] for accession in all_possible],
            )
        )
    )

    # save annotated data
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    data.write(args.output)

 
        