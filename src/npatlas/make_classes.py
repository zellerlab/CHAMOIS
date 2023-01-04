import argparse
import itertools
import json
import gzip
import os

import anndata
import joblib
import numpy
import pandas
import pronto
import rich.progress
import scipy.sparse

# get paths from command line
parser = argparse.ArgumentParser()
parser.add_argument("--atlas", required=True)
parser.add_argument("--chemont", required=True)
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()


# --- Load ChemOnt ------------------------------------------------------------

chemont = pronto.Ontology(args.chemont)
rich.print(f"[bold green]{'Loaded':>12}[/] {len(chemont)} terms from ChemOnt")

chemont_indices = { 
    term.id: i
    for i, term in enumerate(sorted(chemont.terms()))
    if term.id != "CHEMONTID:9999999"
}

# --- Make adjacency matrix for the class graph ------------------------------

superclasses = scipy.sparse.dok_matrix((len(chemont_indices), len(chemont_indices)), dtype=numpy.bool_)
for term_id, i in chemont_indices.items():
    for superclass in chemont[term_id].superclasses():
        if superclass.id != "CHEMONTID:9999999":
            j = chemont_indices[superclass.id]
            superclasses[i, j] = True

subclasses = scipy.sparse.dok_matrix((len(chemont_indices), len(chemont_indices)), dtype=numpy.bool_)
for term_id, i in chemont_indices.items():
    for subclass in chemont[term_id].subclasses():
        if subclass.id != "CHEMONTID:9999999":
            j = chemont_indices[subclass.id]
            subclasses[i, j] = True


# --- Load NPAtlas ------------------------------------------------------------

with rich.progress.open(args.atlas, "rb", description=f"[bold blue]{'Unzipping':>12}[/]") as handle:
    data = json.load(gzip.open(handle))

rich.print(f"[bold green]{'Loaded':>12}[/] {len(data)} compounds from NPAtlas")
np_atlas = {entry["npaid"]: entry for entry in data}
np_atlas_indices = {entry["npaid"]:i for i, entry in enumerate(data)}


# --- Binarize NPAtlas classes ------------------------------------------------

classes = scipy.sparse.dok_matrix((len(np_atlas), len(chemont_indices)), dtype=numpy.bool_)

@joblib.delayed
def binarize(compound):
    annotation = compound["classyfire"]
    if annotation is None or "class" not in annotation:
        return
    # get all parents by traversing the ontology transitively
    direct_parents = pronto.TermSet({
        chemont[direct_parent["chemont_id"]] # type: ignore
        for direct_parent in itertools.chain(
            [annotation["kingdom"], annotation["superclass"], annotation["class"], annotation["subclass"], annotation["direct_parent"]],  
            annotation["intermediate_nodes"],
            annotation["alternative_parents"],
        )
        if direct_parent is not None
    })
    all_parents = direct_parents.superclasses().to_set()
    # set label flag is compound belong to a class
    compound_index = np_atlas_indices[compound["npaid"]]
    for parent in all_parents:
        if parent.id != "CHEMONTID:9999999":
            classes[compound_index, chemont_indices[parent.id]] = True

joblib.Parallel(n_jobs=-1, backend="threading")(
    binarize(compound)
    for compound in rich.progress.track(np_atlas.values(), description=f"[bold blue]{'Binarizing':>12}[/]")
)


# --- Store metadata ----------------------------------------------------------

np_classes = anndata.AnnData(
    X=classes.tocsr(),
    var=pandas.DataFrame(
        index=list(chemont_indices),
        data=dict(name=[chemont[id_].name for id_ in chemont_indices]),
    ),
    varp=dict(
        subclasses=subclasses.tocsr(),
        superclasses=superclasses.tocsr(),
    ),
    obs=pandas.DataFrame(
        index=list(np_atlas_indices),
        data=dict(
            compound=[entry["original_name"] for entry in data],
            inchikey=[entry["inchikey"] for entry in data],
            smiles=[entry["smiles"] for entry in data],
        )
    ),
    dtype=numpy.bool_
)

# save annotated data
os.makedirs(os.path.dirname(args.output), exist_ok=True)
np_classes.write(args.output)
