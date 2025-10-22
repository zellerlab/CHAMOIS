import argparse
import json
import os
import pathlib
import sys
import webbrowser

import pyhmmer
import numpy
import pandas
import pronto
import rich.progress
from rich.console import Console
from palettable.cartocolors.qualitative import Bold_10

folder = pathlib.Path(__file__).absolute().parent
PROJECT_FOLDER = folder
while not PROJECT_FOLDER.joinpath("chamois").exists():
    PROJECT_FOLDER = PROJECT_FOLDER.parent
sys.path.insert(0, str(PROJECT_FOLDER))
from chamois.predictor import ChemicalOntologyPredictor

parser = argparse.ArgumentParser()
parser.add_argument("--pfam", required=True, type=pathlib.Path)
parser.add_argument("--chemont", required=True, type=pathlib.Path)
parser.add_argument("--ec-domain", required=True, type=pathlib.Path)
parser.add_argument("--model", required=True, type=pathlib.Path)
parser.add_argument("--show", action="store_true")
args = parser.parse_args()

console = Console()

# --- Load Pfam ---------------------------------------------------------------

console.print(f"[bold blue]{'Loading':>12}[/] Pfam HMMs")
with rich.progress.open(args.pfam, "rb", description=f"[bold blue]{'Loading':>12}[/]") as f:
    with pyhmmer.plan7.HMMFile(f) as hmm_file:
        pfam_names = {hmm.accession.decode():hmm.name.decode() for hmm in hmm_file}

# --- Load ChemOnt ------------------------------------------------------------

chemont = pronto.Ontology(args.chemont)
console.print(f"[bold green]{'Loaded':>12}[/] {len(chemont)} terms from ChemOnt")

chemont_indices = { 
    term.id: i
    for i, term in enumerate(sorted(chemont.terms()))
    if term.id != "CHEMONTID:9999999"
}

chemont_depth = {
    term.id: len(term.superclasses().to_set())
    for term in chemont.terms()
    if term.id != "CHEMONTID:9999999" 
}
max_chemont_depth = max(chemont_depth.values())

organic_superclasses = chemont["CHEMONTID:0000000"].subclasses(with_self=False, distance=1).to_set()
chemont_superclass = {
    term.id:c
    for c in organic_superclasses
    for term in c.subclasses()
}

# --- Load EC-domains annotations ----------------------------------------------

console.print(f"[bold blue]{'Loading':>12}[/] EC-Pfam association")
assoc = pandas.read_table(args.ec_domain)
ecs = { row["Pfam-Domain"]:row["EC-Number"] for _, row in assoc.iterrows() } 
TOP_LEVEL_ECS = { '1': "Oxidoreductase", '2': "Transferase", '3': "Hydrolase", '4': "Lyase", '5': "Isomerase", '6': "Ligase", '7': "Translocase" }

# --- Load CHAMOIS model ---------------------------------------------------------

console.print(f"[bold blue]{'Loading':>12}[/] trained predictor")
with args.model.open("rb") as f:
    predictor = ChemicalOntologyPredictor.load(f)
coef = predictor.coef_.toarray()

# --- Record domain contribution -----------------------------------------------

rows = []

for i in range(len(predictor.classes_)):

    weights = coef[:, i]
    indices = weights.argsort()
    
    pos_indices = indices[ weights[indices] > 0 ]
    neg_indices = indices[ weights[indices] < 0 ]

    if not len(pos_indices) or not len(neg_indices):
        continue
    
    pos_domain = predictor.features_.index[pos_indices[-1].item()]
    pos_name = predictor.features_.name[pos_domain]
    
    neg_domain = predictor.features_.index[neg_indices[0].item()]
    neg_name = predictor.features_.name[neg_domain]

    id_ = predictor.classes_.index[i]
    name = predictor.classes_.name[i]
    
    rows.append([
        id_, 
        name, 
        len(indices), 
        len(pos_indices), 
        len(neg_indices), 
        pos_domain, 
        pos_name, 
        weights[pos_indices[-1]].item(), 
        neg_domain, 
        neg_name, 
        weights[neg_indices[0]].item(),
        predictor.intercept_[i].item()
    ])
    
df2 = pandas.DataFrame(
    rows, 
    columns=[
        "class_id", 
        "class_name", 
        "features", 
        "pos_features", 
        "neg_features", 
        "pos_domain", 
        "pos_domain_name", 
        "pos_domain_weight", 
        "neg_domain", 
        "neg_domain_name", 
        "neg_domain_weight",
        "intercept",
    ]
)
df2.to_csv(folder.joinpath("domain_contribution.tsv"), sep="\t", index=False)
df2


# --- Compute normalized coefficients ------------------------------------------

# normalize coefficients
norm_coefs = numpy.zeros((len(predictor.classes_), len(predictor.features_)))
for i in range(len(predictor.classes_)):
    pos_coefficients = coef[:, i].clip(min=0)
    neg_coefficients = coef[:, i].clip(max=0)
    pos_normalized = pos_coefficients / numpy.linalg.norm(pos_coefficients)
    neg_normalized = neg_coefficients / numpy.linalg.norm(neg_coefficients)   
    norm_coefs[i] = pos_normalized + neg_normalized

# --- Generate Vega graph ------------------------------------------------------

TOP = 2
N_CLASSES = len(predictor.classes_)
THRESHOLD = 0
name_index = {}

with folder.joinpath("spec.vega.json").open() as f:
    vega_graph = json.load(f)

# generate nodes for classes
vega_graph["data"][0]["values"] = nodes = []
for i, classname in enumerate(predictor.classes_.index):
    if predictor.intercept_[i] > 0:
        continue
    name_index[classname] = len(name_index)
    nodes.append({
        "id": classname,
        "name": chemont[classname].name,
        "index": name_index[classname],
        "organic_superclass": chemont_superclass[classname].name,
        "type": "class",
        "depth": chemont_depth[classname],
        "size": max_chemont_depth - chemont_depth[classname] + 1,
    })
    
# count number of links per domain
n_links = numpy.zeros(len(predictor.features_))
for i, classname in enumerate(predictor.classes_.index):
    if predictor.intercept_[i] > 0:
        continue
    sorted_indices = { x:n for n,x in enumerate(reversed(coef[:, i].argsort())) }
    for j, name in enumerate(predictor.features_.index):
        if norm_coefs[i, j] > THRESHOLD and sorted_indices[j] < TOP:
            n_links[j] += 1
for j, feat in enumerate(predictor.features_.index):
    if n_links[j] > 0:
        name_index[feat] = len(name_index)
        ec = ecs.get(feat.rsplit(".", 1)[0], "Unknown")
        nodes.append({
            "id": feat,
            "name": pfam_names.get(feat, feat),
            "index": name_index[feat],
            "ec": ec,
            "ec_kind": "Unknown" if ec == "Unknown" else TOP_LEVEL_ECS[ ec.split('.')[0] ],
            "type": "domain",
            "size": 8,
            #"depth": 1,
        })
    
# generate edges
vega_graph["data"][1]["values"] = edges = []
for i, classname in enumerate(predictor.classes_.index):
    if predictor.intercept_[i] > 0:
        continue
    sorted_indices = { x:n for n,x in enumerate(reversed(coef[:, i].argsort())) }
    for j, name in enumerate(predictor.features_.index):
        if norm_coefs[i, j] > THRESHOLD and sorted_indices[j] < TOP:
            edges.append({
                "source": name_index[name],
                "target": name_index[classname],
                "weight": 1, #predictor.coef_[j, i],
                "rank": sorted_indices[j] + 1,               
            })
    
# remove nodes without links
# for node in nodes.copy():
#     if node["type"] == "domain":
#         if n_links[node["index"]] == 0:
#             nodes.remove(node)
    
# remove features
with folder.joinpath("graph.html").open("w") as f:
    spec = json.dumps(vega_graph, indent=4, sort_keys=True)
    f.write(
        f"""
        <head>
          <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
          <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
          <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
        </head>
        <body>
          <div id="view"></div>
          <script>
            vegaEmbed( '#view', {spec} );
          </script>
        </body>
        """
    )

# show plot
if args.show:
    webbrowser.open(os.path.abspath("graph.html"))
