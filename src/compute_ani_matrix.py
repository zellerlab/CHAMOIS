import argparse
import statistics
import tarfile
import os

import pyfastani
import gb_io
import rich.progress
import scipy.sparse


parser = argparse.ArgumentParser()
parser.add_argument("--gbk", required=True)
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
                        records[record.name] = record

    # lookup table for BGC name to index
    bgc_indices = { name: i for i, name in enumerate(sorted(records)) }

    # sketch sequences
    average_size = statistics.mean(len(record.sequence) for record in records.values())
    sketch = pyfastani.Sketch(fragment_length=400, percentage_identity=50.0, reference_size=average_size)
    for bgc_id, bgc in progress.track(records.items(), description="Sketching..."):
        sketch.add_genome(bgc.name, bgc.sequence)
    
    progress.console.print("Indexing...")
    mapper = sketch.index()

    # prepare identity matrix
    identity = scipy.sparse.dok_matrix((len(records), len(records)))

    # compute ANI
    for bgc_id, bgc in progress.track(records.items(), description="Mapping..."):
        for hit in mapper.query_genome(bgc.sequence):
            i = bgc_indices[bgc_id]
            j = bgc_indices[hit.name]
            identity[i, j] = identity[j, i] = hit.identity / 100.0

    # save identity matrix
    progress.console.print("Saving...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as dst:
        scipy.sparse.save_npz(dst, identity.tocoo())
