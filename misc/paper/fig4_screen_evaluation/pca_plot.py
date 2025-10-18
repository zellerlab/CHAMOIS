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
import Bio.SeqIO
import pandas
import numpy
import rich.progress
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from palettable.cartocolors.qualitative import Bold_10
from sklearn.decomposition import PCA, SparsePCA

rcParams['svg.fonttype'] = 'none'
folder = pathlib.Path(__file__).absolute().parent
sys.path.insert(0, str(folder.parents[1]))

import chamois.classyfire
from chamois.predictor import ChemicalOntologyPredictor

random.seed(0)


TYPE_PALETTE = {
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

# --- Load data ----------------------------------------------------------------

# load CHAMOIS model
predictor = ChemicalOntologyPredictor.trained()

# load NPAtlas
npatlas = anndata.read(folder.parents[1].joinpath("data", "npatlas", "classes.hdf5"))
npatlas = npatlas[:, predictor.classes_.index]

# load classification of native BGCs
native = anndata.read_h5ad(folder.parents[1].joinpath("data", "datasets", "native", "classes.hdf5"))
native = native[:, predictor.classes_.index]
native.obs["genome_id"] = native.obs_names.str.rsplit("_").str[0]

# --- Get types of true clusters -----------------------------------------------

# load antiSMASH type translation map
type_map = pandas.read_table(folder.joinpath("chem_class_map.tsv"))
type_index = dict(zip(type_map["chem_code"].str.lower(), type_map["bigslice_class"]))
mibig_types = {}

# load coordinates of native BGCs
coordinates = pandas.read_table(
    folder.parents[1].joinpath("data", "datasets", "native", "coordinates.tsv")
)

# load classification of native BGCs
classes = anndata.read_h5ad(
    folder.parents[1].joinpath("data", "datasets", "native", "classes.hdf5")
)

# load CHAMOIS probabilities
probas = anndata.read(folder.joinpath("merged.hdf5"))

# load merged clusters coordinates
merged = pandas.read_table(folder.joinpath("merged.tsv"))
merged['color'] = merged['type'].apply(lambda x: TYPE_PALETTE["Mixed"] if ";" in x else TYPE_PALETTE[x])

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

# extract distances by genome
class GenomePlot(typing.NamedTuple):
    genome: str
    compound: str
    Y: numpy.ndarray
    C: numpy.ndarray
    T: numpy.ndarray

    @property
    def ranks(self):
        return scipy.stats.rankdata(-self.Y, method="dense")

    @property
    def relative_ranks(self):
        return self.ranks / self.Y.shape[0]

genomes = []

# get colors and native BGCs to print
native_compounds = []
for i, row in enumerate(rich.progress.track(coordinates.itertuples(), total=len(coordinates))):
    if classes.obs["unknown_structure"].loc[row.bgc_id]:
        rich.print(f"[bold yellow]{'Skipping':>12}[/] BGC {row.bgc_id!r} with unknown classification")
        continue
    # compounds[i] = classes[row.bgc_id, predictor.classes_.index].X.toarray()[0]
    
    # get probabilities for genome
    genome_clusters = merged[merged.genome_id == row.genome]
    genome_probas = probas[probas.obs_names.isin(set(genome_clusters.cluster_id)), predictor.classes_.index]
    
    # prepare plot
    try:
        true_cluster = genome_probas.obs_names.get_loc(row.true_cluster)
    except KeyError:
        continue # ignore genomes without true cluster in predictions
    native_compounds.append(row.compound)


# --- PCA ----------------------------------------------------------------------

# get colors
types = pandas.read_table(folder.parents[1].joinpath("data", "datasets", "native", "types.tsv"))
types['color'] = types['type'].apply(lambda x: TYPE_PALETTE["Mixed"] if ";" in x else TYPE_PALETTE[x])
colors = { row.bgc_id:row.color for row in types.itertuples()  }

# make PCA
pca = PCA(n_components=2)
npatlas_pca = pca.fit_transform(npatlas.X.toarray())

# transform native BGCs
native = native[native.obs["compound"].isin(native_compounds)]
native_pca = pca.transform(native.X.toarray())

plt.figure(figsize=(12, 6))
plt.scatter(npatlas_pca[:, 0], npatlas_pca[:, 1], color="gray", alpha=0.2)

native_colors = set()
for i, row in enumerate(native.obs.itertuples()):
    bgc_types = sorted(types[types.bgc_id == row.Index]['type'].values[0].split(";"))
    if len(bgc_types) >= 2:
        c1, c2 = map(TYPE_PALETTE.__getitem__, bgc_types)
        native_colors.add(c1)
        native_colors.add(c2)
        plt.scatter(native_pca[i, 0], native_pca[i, 1], c=c1, marker=MarkerStyle("o", fillstyle="right"), edgecolors="black", s=rcParams['lines.markersize']**2.5)
        plt.scatter(native_pca[i, 0], native_pca[i, 1], c=c2, marker=MarkerStyle("o", fillstyle="left"), edgecolors="black", s=rcParams['lines.markersize']**2.5)
    else:
        c = TYPE_PALETTE[bgc_types[0]]
        plt.scatter(native_pca[i, 0], native_pca[i, 1], c=c, edgecolors="black", s=rcParams['lines.markersize']**2.5)
        native_colors.add(c)

legend_elements = [
    Line2D([0], [0], linestyle='', marker='o', markerfacecolor=v, label=k, markeredgecolor="black") #, s=rcParams['lines.markersize']**2.5)
    for k,v in TYPE_PALETTE.items()
    if v in native_colors
]

plt.legend(handles=legend_elements, loc="upper left")
plt.xlabel("Principal component 1 ({:.1%})".format(pca.explained_variance_ratio_[0]))
plt.ylabel("Principal component 2 ({:.1%})".format(pca.explained_variance_ratio_[1]))
plt.tight_layout()
plt.savefig(folder.joinpath("pca.svg"))
plt.savefig(folder.joinpath("pca.png"))
plt.show()
