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
import gb_io

parser = argparse.ArgumentParser()
parser.add_argument("-q", "--query", required=True)
parser.add_argument("-t", "--target", required=True)
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()

with rich.progress.Progress() as progress:

    # load query records
    with progress.open(args.query, "rb") as src:
        query_records = {
            record.name:record
            for record in gb_io.iter(src)
        }
        query_ids = sorted(query_records)

    # load target records
    with progress.open(args.target, "rb") as src:
        target_records = {
            record.name:record
            for record in gb_io.iter(src)
        }
        target_ids = sorted(target_records)

    # create matrix identity
    identity = scipy.sparse.dok_matrix((len(query_records), len(target_records)), dtype=numpy.float)

    # compute ANI
    for j, target_id in enumerate(progress.track(target_ids, description="Computing ANI...")):
        sketch = pyfastani.Sketch(fragment_length=500, reference_size=100000, percentage_identity=50.0)
        sketch.add_genome( target_id, target_records[target_id].sequence )
        mapper = sketch.index()
        for i, query_id in enumerate(progress.track(query_ids, description="Mapping...")):
            hit = next(iter(mapper.query_genome(query_records[query_id].sequence)), None)
            if hit is not None:
                identity[i, j] = hit.identity / 100

    # generate annotated data
    data = anndata.AnnData(
        dtype=numpy.float,
        X=counts.tocsr(),
        obs=pandas.DataFrame(index=query_ids),
        var=pandas.DataFrame(index=target_ids),
    )

    # save annotated data
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    data.write(args.output)
