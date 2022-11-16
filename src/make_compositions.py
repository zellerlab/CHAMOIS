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

import numpy
import pubchempy
import scipy.sparse
import gb_io
import rich.progress
import pyhmmer
import pyrodigal
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

parser = argparse.ArgumentParser()
parser.add_argument("--hmm", required=True)
parser.add_argument("--gbk", required=True)
parser.add_argument("--json", required=True)
parser.add_argument("--p-value", type=float, default=1e-9)
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()

with rich.progress.Progress() as progress:

    # load MIBIG 3 records
    records = {}
    with progress.open(args.gbk, "rb", description="Reading...") as f:
        with tarfile.open(fileobj=f) as tar:
            for entry in iter(tar.next, None):
                if entry.name.endswith(".gbk"):
                    with tar.extractfile(entry) as f:
                        record = next(gb_io.iter(f))
                        records[record.name] = SeqRecord(
                            id=record.name,
                            name=record.name,
                            seq=Seq(record.sequence)
                        )

    # load MIBIG 3 metadata
    mibig = {}
    with tarfile.open(args.json) as tar:
        for entry in iter(tar.next, None):
            if entry.name.endswith(".json"):
                with tar.extractfile(entry) as f:
                    data = json.load(f)
                mibig[data["cluster"]["mibig_accession"]] = data

    # extract contigs of active entries
    contigs = {
        bgc_id: records[bgc_id]
        for bgc_id, bgc in mibig.items()
        if bgc["cluster"]["status"] == "active"
    }

    # find ORFs
    orf_finder = pyrodigal.OrfFinder(meta=True)
    task = progress.add_task(total=len(contigs), description="Finding ORFs...")
    def find_genes(bgc_id, contig):
        genes = orf_finder.find_genes(str(contig.seq))
        progress.update(task_id=task, advance=1)
        return bgc_id, genes
    with multiprocessing.pool.ThreadPool() as pool:
        genes = dict(pool.starmap(find_genes, contigs.items()))

    # convert ORFs to PyHMMER sequences
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
    with gzip.open(args.hmm) as src:
        with pyhmmer.plan7.HMMFile(src) as hmm_file:
            all_possible = sorted( hmm.accession.decode() for hmm in hmm_file )
            name_index = { name: i for i,name in enumerate(all_possible) }

    # annotate genes with Pfam domains
    all_sequences = list(itertools.chain.from_iterable(sequences.values()))
    domain_counts = collections.defaultdict(collections.Counter)
    with gzip.open(args.hmm) as src:
        with pyhmmer.plan7.HMMFile(src) as hmm_file:
            task = progress.add_task(total=len(all_possible), description="Finding domains...")
            callback = lambda *args: progress.update(task_id=task, advance=1)
            for hits in pyhmmer.hmmer.hmmsearch(hmm_file, all_sequences, callback=callback):
                hmm_accession = hits.query_accession.decode()
                for hit in hits:
                    if hit.pvalue < args.p_value:
                        sequence_id = hit.name.decode()
                        bgc_id = sequence_id.split("_", 1)[0]
                        domain_counts[bgc_id][hmm_accession] += 1
            
    # compute domain counts
    bgc_ids = sorted(bgc_id for bgc_id in mibig)
    counts = numpy.zeros((len(bgc_ids), len(all_possible)))
    for i, bgc_id in enumerate(bgc_ids):
        for domain_name, domain_count in domain_counts[bgc_id].items():
            j = name_index[domain_name]
            counts[i, j] = domain_count

    # compute frequencies
    compositions = numpy.nan_to_num(counts / counts.sum(axis=1).reshape(-1, 1))

    # save results
    os.makedirs(args.output, exist_ok=True)
    scipy.sparse.save_npz(os.path.join(args.output, "counts.npz"), scipy.sparse.coo_matrix(counts))
    scipy.sparse.save_npz(os.path.join(args.output, "compositions.npz"), scipy.sparse.coo_matrix(compositions))
    with open(os.path.join(args.output, "domains.tsv"), "w") as out:
        out.writelines(f"{domain}\n" for domain in all_possible)
    with open(os.path.join(args.output, "labels.tsv"), "w") as out:
        out.writelines(f"{label}\n" for label in bgc_ids)
        