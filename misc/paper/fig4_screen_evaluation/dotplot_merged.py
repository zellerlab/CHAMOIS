import pathlib
import random
import io
import os
import zipfile
import json
import sys
import time
import typing
import random

import anndata
import gb_io
import Bio.SeqIO
import pandas
import numpy
import pyskani
import rdkit.Chem
import rich.progress
import scipy.stats
import scipy.sparse
import scipy.spatial.distance
import matplotlib.pyplot as plt
from rdkit import RDLogger
from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder
from matplotlib import rcParams
from matplotlib.markers import MarkerStyle
from palettable.cartocolors.qualitative import Bold_10, Vivid_2

rcParams['svg.fonttype'] = 'none'
folder = pathlib.Path(__file__).absolute().parent
sys.path.insert(0, str(folder.parents[2]))

import chamois.classyfire
from chamois.predictor import ChemicalOntologyPredictor

random.seed(0)
MIBIG_VERSION = "4.0"

# disable logging
RDLogger.DisableLog('rdApp.warning')

def probjaccard(x: numpy.ndarray, y: numpy.ndarray, mask: typing.Optional[numpy.ndarray] = None) -> float:
    if mask is not None:
        x = x[mask]
        y = y[mask]
    tt = x * y
    return (tt).sum() / ((x + y - tt)).sum()

def jaccard(x: numpy.ndarray, y: numpy.ndarray, mask: typing.Optional[numpy.ndarray] = None, t: typing.Optional[numpy.ndarray] = None) -> float:
    if mask is not None:
        x = x[mask]
        y = y[mask]
        if t is not None:
            t = t[mask]
    if t is None:
        t = 0.5
    x = x > t
    y = y > t
    return (x & y).sum() / (x | y).sum()

# PALETTE = dict(zip(
#     ["Polyketide", "NRP", "RiPP", "Saccharide", "Terpene", "Alkaloid", "Mixed", "Other", "Unknown"],
#     ["#1e88e5", "#884ea0",  "#fdd835", "#ec407a", "#009688", "#ef6c00", Bold_10.hex_colors[-3], "#607d8b", "#bebebe"],
# ))
PALETTE = {
    'Polyketide': '#1e88e5',
    'NRP': '#884ea0',
    'RiPP': '#fdd835',
    'Saccharide': '#ec407a',
    'Terpene': '#009688',
    'Alkaloid': '#ef6c00',
    'Other': '#607d8b',
    'Mixed': '#80BA5A',
    'Unknown': '#bebebe',
}

# --- Prepare skani database ---------------------------------------------------

# create a skani database for MIBiG 4.0 clusters 
mibig_clusters = folder.parents[2].joinpath("data", "datasets", f"mibig{MIBIG_VERSION}", "clusters.gbk")
database = pyskani.Database(compression=10, marker_compression=50)
with rich.progress.open(mibig_clusters, "rb", description=f"[bold blue]{'Sketching':>12}[/] reference BGCs") as f:
    for record in gb_io.iter(f):
        database.sketch(record.name, record.sequence)

# --- Load data ----------------------------------------------------------------

# load antiSMASH type translation map
type_map = pandas.read_table(folder.joinpath("chem_class_map.tsv"))
type_index = dict(zip(type_map["chem_code"].str.lower(), type_map["bigslice_class"]))
mibig_types = {}

# load coordinates of native BGCs
coordinates = pandas.read_table(
    folder.parents[2].joinpath("data", "datasets", "native", "coordinates.tsv")
)

# load classification of native BGCs
classes = anndata.read_h5ad(
    folder.parents[2].joinpath("data", "datasets", "native", "classes.hdf5")
)
classes = classes[~classes.obs.unknown_structure]

# load classification of native BGCs
features = anndata.read_h5ad(
    folder.parents[2].joinpath("data", "datasets", "native", "features.hdf5")
)
features = features[classes.obs_names]

#
coordinates = coordinates[coordinates.bgc_id.isin(set(classes.obs_names))]
print(len(coordinates))

# load CHAMOIS probabilities
probas = anndata.read_h5ad(folder.joinpath("merged.hdf5"))

# load merged clusters sequences
with rich.progress.open(folder.joinpath("merged.gbk"), "rb", description=f"[bold blue]{'Loading':>12}[/] predicted BGCs") as f:
    merged_clusters = { record.name: record for record in gb_io.load(f) }

# load merged clusters coordinates
merged = pandas.read_table(folder.joinpath("merged.tsv"))
merged['color'] = merged['type'].apply(lambda x: PALETTE["Mixed"] if ";" in x else PALETTE[x])

print(merged["genome_id"].value_counts().describe())

# --- Load cross-validation ----------------------------------------------------

# cv = anndata.read_h5ad(folder.parents[0].joinpath("fig2_cross_validation", "cv.probas.hdf5"))
# cv_classes = anndata.read_h5ad(
#     folder.parents[2].joinpath("data", "datasets", f"mibig{MIBIG_VERSION}", "classes.hdf5")
# )
# cv_classes = cv_classes[cv.obs_names, cv.var_names]

# beta = 1
# import sklearn.metrics

# fbetas = numpy.full(cv.n_vars, 0.0)
# thresholds = numpy.full(cv.n_vars, 0.5)
# for j, name in enumerate(cv.var_names):
#     pr, rc, t = sklearn.metrics.precision_recall_curve(cv_classes.obs_vector(name), cv.obs_vector(name))
#     fbeta = (1 + beta) * (pr * rc) / (beta * pr + rc + 1e-8)
#     thresholds[j] = t[ fbeta.argmax() ]
#     fbetas[j] = fbeta.max()

# --- Get types of true clusters -----------------------------------------------

# find the true BGC among merged predictions
coordinates['true_cluster'] = None
for row in rich.progress.track(coordinates.itertuples(), total=len(coordinates)):
    # get all predictions for sequence
    sequence_clusters = merged[merged.sequence_id == row.sequence_id]
    # find cluster overlapping true BGC
    clusters = sequence_clusters[(sequence_clusters.start <= row.end) & (row.start <= sequence_clusters.end)]
    if len(clusters) == 1:
        coordinates.loc[row.Index, 'true_cluster'] = clusters.cluster_id.values[0]
    elif len(clusters) > 1:
        raise ValueError("More than one true cluster found!")

with_true_cluster = coordinates[~coordinates['true_cluster'].isnull()]
# print(merged[merged.genome_id.isin(with_true_cluster['genome'])])


# extract distances by genome
class GenomePlot(typing.NamedTuple):
    genome: str
    compound: str
    true_cluster: str
    Y: numpy.ndarray   # prob-jaccard
    C: numpy.ndarray   # color
    T: numpy.ndarray   # is true cluster?
    PT: typing.List    # predicted type(s)

    @property
    def ranks(self):
        return scipy.stats.rankdata(-self.Y, method="dense")

    @property
    def relative_ranks(self):
        return self.ranks / self.Y.shape[0]

# compute distances to predictions
genomes = []
with folder.joinpath(f"predictor.mibig{MIBIG_VERSION}.json").open() as f:
    predictor = ChemicalOntologyPredictor.load(f)
compounds = numpy.zeros((len(coordinates), len(predictor.classes_)))

for i, row in enumerate(rich.progress.track(coordinates.itertuples(), total=len(coordinates))):
    if classes.obs["unknown_structure"].loc[row.bgc_id]:
        rich.print(f"[bold yellow]{'Skipping':>12}[/] BGC {row.bgc_id!r} with unknown classification")
        continue
    compounds[i] = classes[row.bgc_id, predictor.classes_.index].X.toarray()[0]

    # get probabilities for genome
    genome_clusters = merged[merged.genome_id == row.genome]
    genome_probas = probas[probas.obs_names.isin(set(genome_clusters.cluster_id)), predictor.classes_.index]
    # prepare plot
    try:
        true_cluster = genome_probas.obs_names.get_loc(row.true_cluster)
    except KeyError:
        print(row.true_cluster, genome_probas.obs)
        rich.print(f"[bold yellow]{'Skipping':>12}[/] BGC [bold cyan]{row.bgc_id}[/] ([purple]{row.compound}[/]) with no prediction intersecting true BGC")
        continue # ignore genomes without true cluster in predictions
    Y, T, C, PT, types = [], [], [], [], []
    for j in range(len(genome_probas)):
        d = probjaccard(compounds[i], genome_probas.X[j])
        cluster = genome_probas.obs_names[j]
        color = genome_clusters.color[genome_clusters.cluster_id == cluster].values[0]
        T.append(j == true_cluster)
        Y.append(d)
        C.append(color)
        PT.append(merged[ merged.cluster_id == cluster ]['type'].values[0])

    genomes.append(
        GenomePlot(
            genome=row.genome,
            compound=row.compound,
            true_cluster=genome_probas.obs_names[true_cluster],
            Y=numpy.array(Y),
            C=numpy.array(C),
            T=numpy.array(T),
            PT=PT,
        )
    )

# sort results by top true compound
genomes.sort(key=lambda g: tuple(g.Y[g.T]), reverse=True)

# --- Compute top-k scores -----------------------------------------------------

# bold labels for genomes where the true compound was identified
accurate = [ (g.Y[g.T] == g.Y.max()).any() for g in genomes ]
labels = [ g.compound for a, g in zip(accurate, genomes) ]

n_accurate = sum(accurate)
rich.print(f"[bold green]{'Predicted':>12}[/] {sum(accurate)} BGCs accurately out of {len(genomes)} instances ({sum(accurate) / len(genomes) : 3.1%})")

def topN(X, n=5):
    i = X.argsort()
    x = i[-n] if len(i) >= n else i[-1]
    return X[x]

top3_accurate = [ (g.Y[g.T] >= topN(g.Y, n=3)).any() for g in genomes ]
rich.print(f"[bold green]{'Predicted':>12}[/] {sum(top3_accurate)} BGCs somewhat accurately (Top 3) out of {len(genomes)} instances ({sum(top3_accurate) / len(genomes) : 3.1%})")

top5_accurate = [ (g.Y[g.T] >= topN(g.Y, n=5)).any() for g in genomes ]
rich.print(f"[bold green]{'Predicted':>12}[/] {sum(top5_accurate)} BGCs somewhat accurately (Top 5) out of {len(genomes)} instances ({sum(top5_accurate) / len(genomes) : 3.1%})")

top10_accurate = [ (g.Y[g.T] >= topN(g.Y, n=10)).any() for g in genomes ]
rich.print(f"[bold green]{'Predicted':>12}[/] {sum(top10_accurate)} BGCs somewhat accurately (Top 10) out of {len(genomes)} instances ({sum(top10_accurate) / len(genomes) : 3.1%})")

# --- Compute p-value ----------------------------------------------------------

# average rank of CHAMOIS predictions
average_chamois_rank = sum(genome.relative_ranks[numpy.nonzero(genome.T)[0][0]] for genome in genomes) / len(genomes)
best_attainable = sum(genome.relative_ranks.min() for genome in genomes) / len(genomes)
# print(f"{average_chamois_rank=}")
# print(f"{best_attainable=}")

# bootstrap distribution to get p-values
N_BOOTSTRAP = 100
random.seed(0)
better = 1
for _ in rich.progress.track(range(N_BOOTSTRAP)):
    selected_ranks = numpy.zeros(len(genomes))
    for i, genome in enumerate(genomes):
        y = list(genome.Y)
        random.shuffle(y)
        relative_ranks = scipy.stats.rankdata(-numpy.array(y), method="dense") / genome.Y.shape[0]
        selected_ranks[i] = relative_ranks[numpy.nonzero(genome.T)[0][0]] 
    average_rank = selected_ranks.sum() / len(genomes)
    if average_rank <= average_chamois_rank:
        better += 1
rich.print(f"P = {better / N_BOOTSTRAP} (N = {N_BOOTSTRAP})")

# --- Plot MIBiG sequence similarity -------------------------------------------

# plot results
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(14, 8), gridspec_kw={'height_ratios':[4,4,4,1,32]}, sharex=True)

def query_similarity(hit: pyskani.Hit):
    return hit.identity * hit.query_fraction

unique_genomes = set()
X = list(range(len(genomes)))
Y = []
table = rich.table.Table("compound", "cluster", "mibig_bgc", "corrected ANI", "identity", "query fraction", "ref fraction")
for genome in genomes:
    true_cluster = merged_clusters[genome.true_cluster]
    hit = max(database.query(true_cluster.name, true_cluster.sequence), key=query_similarity, default=None)
    Y.append(0.0 if hit is None else query_similarity(hit))
    # names.append("" if hit is None else hit.reference_name)
    unique_genomes.add(genome.true_cluster.split("_")[0])
    table.add_row(
        genome.compound, 
        genome.true_cluster, 
        "n/a" if hit is None else str(hit.reference_name),
        "n/a" if hit is None else str(query_similarity(hit)),
        "n/a" if hit is None else str(hit.identity),
        "n/a" if hit is None else str(hit.query_fraction),
        "n/a" if hit is None else str(hit.reference_fraction),
    )
rich.print(table)
print(len(unique_genomes))

# gray background for top-1 and top-5 predictions
axes[0].bar(X, Y, color="#508189")
axes[0].set_ylim(0, 1)
ylim = axes[0].get_ylim()
for x in range(len(labels)):
    if accurate[x]:
        axes[0].fill_between([x-0.5, x+0.5], ylim[0], ylim[1], color="#c0c0c0", zorder=0)
    elif top5_accurate[x]:
        axes[0].fill_between([x-0.5, x+0.5], ylim[0], ylim[1], color="#e0e0e0", zorder=0)
axes[0].set_ylim(ylim)
axes[0].set_ylabel("Sequence\nSimilarity")

# --- Plot MIBiG compound similarity -------------------------------------------

encoder = MHFPEncoder(2048, 42)

# load MIBiG compounds and encode to SMILES
mibig_classes = anndata.read_h5ad(
    folder.parents[2].joinpath("data", "datasets", f"mibig{MIBIG_VERSION}", "classes.hdf5")
)
mibig_classes = mibig_classes[~mibig_classes.obs.unknown_structure]

rich.print(f"[bold blue]{'Encoding':>12}[/] {mibig_classes.n_obs} MIBiG compounds to MHFP6")
mibig_mhfp6 = numpy.array(encoder.EncodeSmilesBulk(list(mibig_classes.obs.smiles), kekulize=True))

X = list(range(len(genomes)))
Y = []
for genome in rich.progress.track(genomes, description=f"[bold blue]{'Searching':>12}[/] closest MIBiG compound"):
    smiles = classes.obs.smiles[classes.obs.compound == genome.compound][0]
    mhfp6 = numpy.asarray(encoder.EncodeSmiles(smiles))
    d = min(
        scipy.spatial.distance.hamming(mhfp6, mibig_mhfp6[i])
        for i in range(mibig_classes.n_obs)
    )
    Y.append(1 - d)

# gray background for top-1 and top-5 predictions
axes[2].bar(X, Y, color=Vivid_2.hex_colors[1])
axes[2].set_ylim(0, 1)
# axes[1].axhline(0.5, color="gray", linestyle="--")
ylim = axes[2].get_ylim()
for x in range(len(labels)):
    if accurate[x]:
        axes[2].fill_between([x-0.5, x+0.5], ylim[0], ylim[1], color="#c0c0c0", zorder=0)
    elif top5_accurate[x]:
        axes[2].fill_between([x-0.5, x+0.5], ylim[0], ylim[1], color="#e0e0e0", zorder=0)
axes[2].set_ylim(ylim)
axes[2].set_ylabel("Chemical\nSimilarity")

# --- Plot MIBiG domain similarity ---------------------------------------------

# load MIBiG 4.0 features
mibig_features = anndata.read_h5ad(
    folder.parents[2].joinpath("data", "datasets", f"mibig{MIBIG_VERSION}", "features.hdf5")
)
mibig_features = mibig_features[mibig_classes.obs_names]

# transform features
F = numpy.zeros((features.n_obs, mibig_features.n_vars))
for i, row in enumerate(features.X.toarray()):
    for j, x in enumerate(row):
        try:
            k = mibig_features.var_names.get_loc(features.var_names[j])
            F[i, k] = x
        except KeyError:
            pass

# plot
unique_genomes = set()
X = list(range(len(genomes)))
Y = []
table = rich.table.Table("compound", "cluster", "mibig_bgc", "corrected ANI", "identity", "query fraction", "ref fraction")
for genome in genomes:
    true_cluster = merged_clusters[genome.true_cluster]
    obs_name = classes.obs_names[classes.obs.compound == genome.compound][0]
    i = classes.obs_names.get_loc(obs_name)
    m = numpy.nan_to_num(scipy.spatial.distance.cdist(F[i:i+1], mibig_features.X.toarray(), metric="jaccard"), nan=1.0)
    Y.append(1 - m.min())

# gray background for top-1 and top-5 predictions
axes[1].bar(X, Y, color=Vivid_2.hex_colors[0])
axes[1].set_ylim(0, 1)
ylim = axes[1].get_ylim()
for x in range(len(labels)):
    if accurate[x]:
        axes[1].fill_between([x-0.5, x+0.5], ylim[0], ylim[1], color="#c0c0c0", zorder=0)
    elif top5_accurate[x]:
        axes[1].fill_between([x-0.5, x+0.5], ylim[0], ylim[1], color="#e0e0e0", zorder=0)
axes[1].set_ylim(ylim)
axes[1].set_ylabel("Domain\nSimilarity")


# --- Plot search results ------------------------------------------------------

# bold labels for genomes where the true compound was identified
accurate = [ (g.Y[g.T] == g.Y.max()).any() for g in genomes ]
labels = [ g.compound for a, g in zip(accurate, genomes) ]

for i, g in enumerate(rich.progress.track(genomes)):

    X = []
    Y = []
    C = []

    X1 = []
    Y1 = []
    C1 = []
    C2 = []

    # plot false clusters
    for t, y, c, types in zip(g.T, g.Y, g.C, g.PT):
        types = types.split(";")
        if not t:
            x = i if t else i + random.random()*0.4 - 0.2
            if len(types) >= 2:
                c1, c2 = map(PALETTE.__getitem__, types[:2])
                C1.append(c1)
                C2.append(c2)
                X1.append(x)
                Y1.append(y)
            else:
                # axes[3].scatter(i, y, c=c, **options)
                X.append(x)
                Y.append(y)
                C.append(c)

    axes[4].scatter(X, Y, c=C, alpha=0.5)
    axes[4].scatter(X1, Y1, c=C1, marker=MarkerStyle("o", fillstyle="right"), alpha=0.5)
    axes[4].scatter(X1, Y1, c=C2, marker=MarkerStyle("o", fillstyle="left"), alpha=0.5)

    # plot true cluster last to be on top of the rest
    for t, y, c, types in zip(g.T, g.Y, g.C, g.PT):
        options = dict(s=rcParams["lines.markersize"]**2.5, alpha=1.0, edgecolors="black")
        if t:
            types = types.split(";")
            if len(types) >= 2:
                c1, c2 = map(PALETTE.__getitem__, types[:2])
                axes[4].scatter(i, y, c=c1, marker=MarkerStyle("o", fillstyle="right"), **options)
                axes[4].scatter(i, y, c=c2, marker=MarkerStyle("o", fillstyle="left"), **options)
            else:
                axes[4].scatter(i, y, c=c, **options)


# X = numpy.array([ i if t else i + random.random()*0.4 - 0.2 for i, g in enumerate(genomes) for t in g.T ])
# Y = numpy.array([ y for g in genomes for y in g.Y ])
# C = numpy.array([ c for g in genomes for c in g.C ])
# T = numpy.array([ t for g in genomes for t in g.T ])

# axes[3].scatter(X[~T], Y[~T], alpha=0.5, c=C[~T])
# axes[3].scatter(X[T], Y[T], alpha=1.0, edgecolors="black", c=C[T], s=rcParams['lines.markersize']**2.5)

axes[4].set_ylabel("Probabilistic Jaccard Similarity")
axes[4].set_xticks(range(len(labels)), labels=labels, rotation=90)#, horizontalalignment='right')

for label,color in PALETTE.items():
    axes[4].scatter([], [], c=color, label=label)

# show predictions per genome as text
for i, g in enumerate(genomes):
    j = numpy.argwhere(g.T)[0]
    rank = g.ranks[j][0]
    axes[3].text(i - 0.4, 0, rank, rotation=90)
axes[3].set_axis_off()
# axes[2].set_ylabel("Rank")

# gray background for top-1 and top-5 predictions
ylim = axes[4].get_ylim()
xlim = axes[4].get_xlim()
for x in range(len(labels)):
    if accurate[x]:
        axes[4].fill_between([x-0.5, x+0.5], ylim[0], ylim[1], color="#c0c0c0", zorder=0)
    elif top5_accurate[x]:
        axes[4].fill_between([x-0.5, x+0.5], ylim[0], ylim[1], color="#e0e0e0", zorder=0)
axes[4].set_xlim(xlim[0] + 2.2, xlim[1] - 2.2)
axes[4].set_ylim(ylim)

fig.tight_layout()
plt.savefig(folder.joinpath("dotplot_merged.svg"))
plt.savefig(folder.joinpath("dotplot_merged.png"))
plt.show()

# normality = scipy.stats.normaltest(dist)
# relative_ranks = [genome.relative_ranks[ genome.Y.argmax() ] for genome in genomes]
# print("relative ranks", relative_ranks)
# average_rank = sum(relative_ranks) / len(genomes)
# print("average rank", average_rank)
# pvalue = scipy.stats.norm.cdf(average_rank, numpy.mean(dist), numpy.std(dist))
# print("pvalue?", pvalue)


# exit(0)
# # plot results - transposed
# fig = plt.figure(figsize=(6, 10))
# axs = plt.axes()
# axs.scatter(Y[~T], X[~T], alpha=0.5, c=C[~T])
# axs.scatter(Y[T], X[T], alpha=1.0, edgecolors="black", c=C[T], s=rcParams['lines.markersize']**2.5)
# axs.set_xlabel("Probabilistic Jaccard Similarity")
# axs.set_yticks(range(len(labels)), labels=labels, horizontalalignment='right')
# axs.invert_yaxis()
# # axs.title.set_text("Clusters")

# for label,color in PALETTE.items():
#     axs.scatter([], [], c=color, label=label)
# axs.legend()

# ylim = axs.get_ylim()
# xlim = axs.get_xlim()
# for x in range(len(labels)):
#     if accurate[x]:
#         axs.fill_betweenx([x-0.5, x+0.5], xlim[0], xlim[1], color="#c0c0c0", zorder=0)
#     elif top5_accurate[x]:
#         axs.fill_betweenx([x-0.5, x+0.5], xlim[0], xlim[1], color="#e0e0e0", zorder=0)
# axs.set_xlim(xlim)
# axs.set_ylim(ylim)

# fig.tight_layout()
# # plt.savefig(folder.joinpath("dotplot_merged.transposed.svg"))
# # plt.savefig(folder.joinpath("dotplot_merged.transposed.png"))
# plt.show()
