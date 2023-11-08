import argparse
import itertools

import anndata
import disjoint_set
import gb_io
import pandas
import pyskani
import rich.progress
import scipy.sparse


# get paths from command line
parser = argparse.ArgumentParser()
parser.add_argument("-q", "--query", required=True)
parser.add_argument("-r", "--reference", required=True)
parser.add_argument("-o", "--output", required=True)
parser.add_argument("-s", "--similarity", type=float, default=0.5)
args = parser.parse_args()

# 
ref_names = []
database = pyskani.Database(compression=10, marker_compression=50)
with rich.progress.open(args.reference, "rb", description=f"[bold blue]{'Sketching':>12}[/] reference BGCs") as f:
    for record in  gb_io.iter(f):
        database.sketch(record.name, record.sequence)
        ref_names.append(record.name)

# 
query_names = []
with rich.progress.open(args.query, "rb", description=f"[bold blue]{'Indexing':>12}[/] query BGCs") as f:
    for record in  gb_io.iter(f):
        query_names.append(record.name)

query_name_index = {x:i for i,x in enumerate(query_names)}
ref_name_index = {x:i for i,x in enumerate(ref_names)}
X = scipy.sparse.dok_matrix(( len(query_names), len(ref_names) ))

#
with rich.progress.open(args.query, "rb", description=f"[bold blue]{'Querying':>12}[/] BGCs") as f:
    for record in  gb_io.iter(f):
        for hit in database.query(record.name, record.sequence):
            i = query_name_index[hit.query_name]
            j = ref_name_index[hit.reference_name]
            X[i, j] = hit.identity * max(hit.query_fraction, hit.reference_fraction)

#
group_set = disjoint_set.DisjointSet({ i:i for i in range(len(query_names)) })
indices = itertools.combinations(range(len(query_names)), 2)
total = len(query_names) * (len(query_names) - 1) / 2
for (i, j) in rich.progress.track(indices, total=total, description=f"[bold blue]{'Grouping':>12}[/]"):
    if X[i, j] > args.similarity:
        group_set.union(i, j)
n = sum(1 for _ in group_set.itersets())
rich.print(f"[bold green]{'Built':>12}[/] {n} groups of clusters with >{args.similarity:.1%} local identity")

rich.print(f"[bold blue]{'Writing':>12}[/] ANI table to {args.output!r}")
obs = pandas.DataFrame(index=query_names, data=dict(groups=[group_set[i] for i in range(len(query_names))]))
var = pandas.DataFrame(index=ref_names)
data = anndata.AnnData(obs=obs, var=var, X=X.tocsr())
data.write(args.output)
