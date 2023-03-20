import argparse

import pandas
import pyhmmer
import rich.progress

parser = argparse.ArgumentParser()
parser.add_argument("--hmm", required=True)
parser.add_argument("--list", required=True)
args = parser.parse_args()

table = pandas.read_csv(args.list, sep="\t", index_col="knum")

hmms = []
with pyhmmer.plan7.HMMFile(args.hmm) as hmm_file:
    for hmm in rich.progress.track(hmm_file, total=len(table)):
        knum = hmm.name.decode()
        threshold = table.loc[knum, "threshold"]
        if threshold != "-":
            threshold = float(threshold)
            hmm.cutoffs.trusted = threshold, threshold
            hmm.accession = hmm.name
            hmm.description = table.loc[knum, "definition"].encode()
            hmms.append(hmm)

with open(args.hmm, "wb") as dst:
    for hmm in hmms:
        hmm.write(dst)
