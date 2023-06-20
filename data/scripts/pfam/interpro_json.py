import argparse
import json
import os
from xml.etree import ElementTree as etree

import pronto
import rich.progress

try:
    from isal import igzip as gzip
except ImportError:
    import gzip

parser = argparse.ArgumentParser()
parser.add_argument("--go", required=True)
parser.add_argument("--xml", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

rich.print(f"[bold blue]{'Loading':>12}[/] Gene Ontology from {args.go!r}")
go = pronto.Ontology(args.go)
top_functions = go['GO:0003674'].subclasses(with_self=False, distance=1).to_set()

rich.print(f"[bold blue]{'Loading':>12}[/] InterPro from {args.xml!r}")
with rich.progress.open(args.xml, mode="rb", description=f"[bold blue]{'Reading':>12}[/]") as src:
    with gzip.GzipFile(fileobj=src, mode="rb") as src:
        tree = etree.parse(src)

# build entries
rich.print(f"[bold blue]{'Building':>12}[/] InterPro entries")
entries = []
for elem in tree.findall("interpro"):
    # extract InterPro name and accession
    accession = elem.attrib["id"]
    name = elem.find("name").text

    # extract Pfam accession, or skip entry if no Pfam member
    for member in elem.find("member_list").iterfind("db_xref"):
        if member.attrib["dbkey"].startswith("PF"):
            break
    else:
        continue

    # extract GO terms
    go_terms = pronto.TermSet()
    class_list = elem.find("class_list")
    if class_list is not None:
        for classif in class_list.iterfind("classification"):
            if classif.attrib["class_type"] == "GO":
                go_terms.add(go[classif.attrib["id"]])

    # save the entry
    entries.append({
        "accession": member.attrib["dbkey"],
        "interpro": accession,
        "name": name,
        "type": elem.attrib["type"].lower(),
        "go_functions": [
            {"accession": term.id, "name": term.name}
            for term in sorted(go_terms.superclasses().to_set() & top_functions)
        ],
        "go_terms": [
            {"accession": term.id, "name": term.name, "namespace": term.namespace}
            for term in sorted(go_terms)
        ],
    })

# sort by id and save
rich.print(f"[bold blue]{'Saving':>12}[/] entries to {args.out!r}")
entries.sort(key=lambda entry: entry["accession"])
with open(args.out, "wt") as dest:
    json.dump(entries, dest, sort_keys=True, indent=4)