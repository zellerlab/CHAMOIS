import argparse
import urllib.request
import os
import io
import tarfile

import rich.progress
import pandas
import Bio.SeqIO


parser = argparse.ArgumentParser()
parser.add_argument("--blocklist")
parser.add_argument("--mibig-version", default="3.1")
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()

url = f"https://dl.secondarymetabolites.org/mibig/mibig_gbk_{args.mibig_version}.tar.gz"

with rich.progress.Progress() as progress:

    # load blocklist if any
    if args.blocklist is not None:
        table = pandas.read_table(args.blocklist)
        blocklist = set(table.bgc_id.unique())
    else:
        blocklist = set()

    # download MIBIG 3 records
    records = []
    with urllib.request.urlopen(url) as response:
        total = int(response.headers["Content-Length"])
        with progress.wrap_file(response, total=total, description="Downloading...") as f:
            with tarfile.open(fileobj=f, mode="r|gz") as tar:
                for entry in iter(tar.next, None):
                    if entry.name.endswith(".gbk"):
                        with tar.extractfile(entry) as f:
                            data = io.StringIO(f.read().decode())
                            record = Bio.SeqIO.read(data, "genbank")
                            # ignore BGCs in blocklist
                            if record.id in blocklist:
                                continue
                            # clamp the BGC boundaries to the left- and rightmost genes
                            start = min( f.location.start for f in record.features if f.type == "CDS" )
                            end = max( f.location.end for f in record.features if f.type == "CDS" )
                            bgc_record = record[start:end]
                            bgc_record.annotations = record.annotations.copy()
                            bgc_record.id = record.id 
                            bgc_record.name = record.name
                            bgc_record.description = record.description
                            records.append(bgc_record)

    # sort records by MIBiG accession
    records.sort(key=lambda record: record.id)

    # save records
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as dst:
        Bio.SeqIO.write(records, dst, "genbank")
