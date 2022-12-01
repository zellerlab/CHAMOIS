import argparse
import contextlib
import statistics
import tarfile
import os
import tempfile

import anndata
import numpy
import pyfastani
import rich.progress
import scipy.sparse
import pandas
import Bio.SeqIO

parser = argparse.ArgumentParser()
parser.add_argument("-q", "--query", required=True)
parser.add_argument("-t", "--target", required=True)
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()

with rich.progress.Progress() as progress:

    # load query records
    with progress.open(args.query, "r") as src:
        query_records = {
            record.id:record
            for record in Bio.SeqIO.parse(src, "genbank")
        }
        query_ids = sorted(query_records)
        query_indices = { name:i for i, name in enumerate(query_ids) }

    # load target records
    with progress.open(args.target, "r") as src:
        target_records = {
            record.id:record
            for record in Bio.SeqIO.parse(src, "genbank")
        }
        target_ids = sorted(target_records)
        target_indices = { name:i for i, name in enumerate(target_ids) }

    # sketch sequences
    average_size = statistics.mean(len(record.seq) for record in target_records.values())
    sketch = pyfastani.Sketch(fragment_length=400, percentage_identity=50.0, reference_size=int(average_size))
    for target_id, target_record in progress.track(target_records.items(), description="Sketching..."):
        sketch.add_genome(target_id, str(target_record.seq))
    
    progress.console.print("Indexing...")
    mapper = sketch.index()

    # create matrix identity
    identity = scipy.sparse.dok_matrix((len(query_records), len(target_records)), dtype=numpy.float_)

    # compute ANI
    for query_id, query_record in progress.track(query_records.items(), description="Mapping..."):
        for hit in mapper.query_genome(str(query_record.seq)):
            i = query_indices[query_id]
            j = target_indices[hit.name]
            identity[i, j] = hit.identity / 100.0

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
