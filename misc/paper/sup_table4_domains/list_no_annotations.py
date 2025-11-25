import collections
import sys
import os
import pathlib
import gzip
import json
import urllib.request

import anndata
import fastobo
import numpy
import pandas
import pyhmmer
import rich.progress
from rich.console import Console

sys.path.insert(0, os.path.realpath(os.path.join(__file__, "..", "..", "..", "..")))
import chamois.predictor

try:
    from lxml import etree
except ImportError:
    from xml.etree import ElementTree as etree

console = Console()

# --- Load ChemOnt -------------------------------------------------------------
chemont = pathlib.Path(__file__).absolute().parents[3].joinpath("data", "ontologies", "ChemOnt_2_1.obo")
ontology = fastobo.load(str(chemont))

parents = {}
for frame in ontology:
    for clause in filter(lambda c: isinstance(c, fastobo.term.IsAClause), frame):
        if str(frame.id) in parents:
            raise RuntimeError(f"Found multiple parents in frame {frame.id!r}")
        parents[str(frame.id)] = str(clause.term)

depths = {}
for term_id in parents:
    depth = 0
    parent = term_id
    while parent in parents:
        parent = parents[parent]
        depth += 1
    depths[term_id] = depth


# --- Parse Pfam ---------------------------------------------------------------

console.print(f"[bold blue]{'Extracting':>12}[/] domain lengths from Pfam")
pfam = pathlib.Path(__file__).absolute().parents[3].joinpath("data", "pfam", "Pfam38.0.hmm")
domain_lengths = {}
with rich.progress.open(pfam, console=console, mode="rb", description=f"[bold blue]{'Reading':>12}[/]") as f:
    with pyhmmer.plan7.HMMFile(f) as hmm_file:
        for hmm in hmm_file:
            domain_lengths[ hmm.accession.decode().rsplit(".", 1)[0] ] = hmm.M

# --- Load EC-domains annotations ----------------------------------------------

console.print(f"[bold blue]{'Loading':>12}[/] EC-Pfam association")
assoc = pandas.read_table(pathlib.Path(__file__).absolute().parents[3].joinpath("data", "ecdomainminer", "EC-Pfam_calculated_associations_Extended.csv"))
ecs = { row["Pfam-Domain"]:row["EC-Number"] for _, row in assoc.iterrows() } 
TOP_LEVEL_ECS = { '1': "Oxidoreductase", '2': "Transferase", '3': "Hydrolase", '4': "Lyase", '5': "Isomerase", '6': "Ligase", '7': "Translocase" }

# --- Download InterPro --------------------------------------------------------

# download InterPro
console.print(f"[bold blue]{'Reading':>12}[/] InterPro metadata")
interpro = pathlib.Path(__file__).absolute().parents[3].joinpath("data", "pfam", "interpro107.0.xml.gz")
with rich.progress.open(interpro, mode="rb", console=console, description=f"[bold blue]{'Reading':>12}[/]") as src:
    with gzip.GzipFile(fileobj=src, mode="rb") as src:
        tree = etree.parse(src)

# build entries
entries = []
for elem in tree.findall("interpro"):
    # extract InterPro name and accession
    accession = elem.attrib["id"]
    name = elem.find("name").text
    # extract Pfam accession, or skip entry if no Pfam member
    members = set()
    databases = set()
    for member in elem.find("member_list").iterfind("db_xref"):
        members.add(member.attrib["dbkey"])
        databases.add(member.attrib["db"])
    if "PFAM" not in databases:
        continue
    # extract EC numbers
    entry_ecs = set()
    xrefs = elem.find("external_doc_list")
    if xrefs is not None:
        for xref in xrefs.iterfind("db_xref"):
            if xref.attrib["db"] == "EC":
                entry_ecs.add(xref.attrib["dbkey"])
    # extract GO terms
    entry_gos = set()
    classlist = elem.find("class_list")
    if classlist is not None:
        for c in classlist.iterfind("classification"):
            if c.attrib["class_type"] == "GO":
                entry_gos.add(c.attrib["id"])
    # save the entry
    entries.append({
        "accession": accession,
        "members": sorted(members),
        "name": name,
        "databases": sorted(databases),
        "type": elem.attrib["type"].lower(),
        "uniprot_protein_count": int(elem.attrib["protein_count"]),
        "ec_numbers": sorted(entry_ecs),
        "go_terms": sorted(entry_gos),
    })
del tree


# --- Load training dataset 

cv = pandas.read_table(
    pathlib.Path(__file__).absolute().parents[1].joinpath("fig2_cross_validation", "cv.report.tsv"),
    index_col="class",
)
classes = anndata.read_h5ad(
    pathlib.Path(__file__).absolute().parents[3].joinpath("data", "datasets", "mibig4.0", "classes.hdf5")
)
features = anndata.read_h5ad(
    pathlib.Path(__file__).absolute().parents[3].joinpath("data", "datasets", "mibig4.0", "features.hdf5")
)

classes = classes[~classes.obs.unknown_structure]
features = features[classes.obs.index]

results = []
has_unannotated_domain = numpy.zeros(features.n_obs, dtype=numpy.bool_)

for i, feature in enumerate(rich.progress.track(features.var.itertuples(), total=len(features.var), description=f"[bold blue]{'Working':>12}[/]")):
    pfam_accession = feature.Index.rsplit(".", 1)[0]
    feature_index = features.var_names.get_loc(feature.Index)
    occurrences = features.obs_vector(feature.Index)
    interpro_entry = next((entry for entry in entries if pfam_accession in entry['members']), {})

    # ignore features with DUF annotation
    if pfam_accession in ecs or interpro_entry.get("ec_numbers") or interpro_entry.get("go_terms"):
        continue
    
    if numpy.any(occurrences > 0):
        print(pfam_accession, feature.name, numpy.sum(occurrences > 0))

    results.append(pfam_accession)
    has_unannotated_domain |= occurrences > 0

print(f"{features.n_obs=}")
print(f"{has_unannotated_domain.sum()=}")

    # bgcs = features.obs_names[(features.X[:, feature_index] > 0).nonzero()[0]]
    # weights = coef_[i]
    # positives = weights > 2.0

    # try:
    #     url = f"https://www.ebi.ac.uk/interpro/api/structure/pdb/entry/pfam/{pfam_accession}"
    #     with urllib.request.urlopen(url) as res:
    #         data = json.load(res)
    #         pdb = data["results"][0]["metadata"]["accession"]
    # except Exception:
    #     pdb = None

    # try:
    #     url = f"https://www.ebi.ac.uk/interpro/api/protein/uniprot/entry/pfam/{pfam_accession}"
    #     with urllib.request.urlopen(url) as res:
    #         data = json.load(res)
    #         uniprot = data["results"][0]["metadata"]["accession"]
    # except Exception:
    #     uniprot = None

    # for class_ in predictor.classes_[positives].itertuples():
    #     # get class index
    #     j = predictor.classes_.index.get_loc(class_.Index)
    #     # get the rank of the DUF among weights
    #     ranks = numpy.argsort(-coef_[:, j])
    #     rank = numpy.argwhere(ranks == i)[0, 0] + 1
    #     # record results
    #     results.append(
    #         [
    #             feature.Index,
    #             feature.name,
    #             occurrences,

    #             rank,
    #             domain_lengths[pfam_accession],

    #             # ";".join(sorted(bgcs)),
    #             class_.Index,
    #             class_.name,
    #             class_.n_positives,
    #             depths[class_.Index],
    #             cv.loc[class_.Index].auprc,
    #             weights[j],

    #             interpro_entry.get("accession"),
    #             interpro_entry.get("name"),
    #             interpro_entry.get("uniprot_protein_count"),
    #             uniprot,
    #             pdb,
    #         ]
    #     )

# out = pandas.DataFrame(
#     results,
#     columns=[
#         "domain_accession",
#         "domain_name",
#         "domain_occurences",
#         "domain_rank",
#         "domain_length",
#         # "domain_bgcs",
#         "class_accession",
#         "class_name",
#         "class_occurences",
#         "class_depth",
#         "cv_auprc",
#         "weight",
#         "interpro_entry",
#         "interpro_description",
#         "uniprot_occurences",
#         "uniprot_accession",
#         "pdb_structure",
#     ],
# )
# out.sort_values(["domain_accession", "weight"], inplace=True)
# out.to_csv(pathlib.Path(__file__).absolute().parent.joinpath("table.tsv"), sep="\t", index=False)
