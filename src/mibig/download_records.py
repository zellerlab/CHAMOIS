import argparse
import urllib.request
import os
import tarfile

import rich.progress
import pandas
import gb_io


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
                            record = next(gb_io.iter(f))
                            if record.name not in blocklist:
                                records.append(record)

    # sort records by name
    records.sort(key=lambda record: record.name)

    # save records
    os.makedirs(os.path.dirname(args.output))
    with open(args.output, "wb") as dst:
        gb_io.dump(records, dst)