import argparse
import urllib.request
import tarfile
import io

import pandas
import pyhmmer

parser = argparse.ArgumentParser()
parser.add_argument("--version", default="6-1-1")
parser.add_argument("--output", required=True)
args = parser.parse_args()

with urllib.request.urlopen(f"https://github.com/antismash/antismash/archive/refs/tags/{args.version}.tar.gz") as res:
    data = io.BytesIO(res.read())

with tarfile.open(fileobj=data) as tar:
    with tar.extractfile(f"antismash-{args.version}/antismash/detection/hmm_detection/hmmdetails.txt") as f:
        table = pandas.read_table(f, header=None, names=["name", "description", "threshold", "filename"])
        table["smcog_id"] = table.index.apply("SMCOGS{:05}".format)
        table = table.set_index("name")
    with open(args.output, "wb") as dst:
        for entry in iter(tar.next, None):
            path = entry.name.split("/")
            if "hmm_detection" in path and entry.name.endswith(".hmm"):
                with tar.extractfile(entry) as f:
                    with pyhmmer.plan7.HMMFile(f) as hmm_file:
                        hmm = hmm_file.read()
                i = table.index.get_loc(hmm.name.decode())
                t = table.iloc[i, "threshold"]
                hmm.name = table.iloc[i, "name"].encode()
                hmm.accession = table.iloc[i, "smcog_id"].encode()
                hmm.description = table.iloc[i, "description"].encode()
                hmm.cutoffs.trusted = t, t
                hmm.write(dst)


