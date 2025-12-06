import argparse
import urllib.error
import urllib.request
import json
import pathlib
import itertools
import io

import Bio.Entrez
import Bio.SeqIO
import rich.progress

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", required=True, type=pathlib.Path)
parser.add_argument("--output", "-o", required=True, type=pathlib.Path)
parser.add_argument("--email", default="martin.larralde@embl.de")
args = parser.parse_args()

Bio.Entrez.email = args.email

with args.input.open() as src:
    entries = tuple(json.load(src))

peptides = []
for entry in rich.progress.track(entries, description=f"[bold blue]{'Working':>12}[/]"):
    accession = entry["accession"]
    enzyme = entry["enzyme"]
    aux = enzyme.get("auxiliaryEnzymes", [])
    result = {
        "accession": entry["accession"],
        "enzymes": []
    }
    for ids in itertools.chain([enzyme["databaseIds"]], [e["databaseIds"] for e in aux]):
        if "uniprot" in ids:
            try:
               db = "uniparc" if ids['uniprot'].startswith("UP") else "uniprotkb"
               url = f"https://rest.uniprot.org/{db}/{ids['uniprot']}.fasta"
               with urllib.request.urlopen(url) as res:
                   seq = Bio.SeqIO.read(io.TextIOWrapper(res), "fasta")
            except Exception as err:
                rich.print(f"[bold red]{'Error':>12}[/] downloading from {url}: {err}")
            else:
                result["enzymes"].append({ "ids": ids, "sequence": str(seq.seq) })
                continue

        if "genpept" in ids:
            try:
                with Bio.Entrez.efetch(db="protein", id=ids['genpept'], rettype="fasta", retmode="text") as src:
                    seq = Bio.SeqIO.read(src, "fasta")
            except Exception as err:
                rich.print(f"[bold red]{'Error':>12}[/] downloading from {ids['genpept']} from GenPept: {err}")
            else:
                result["enzymes"].append({ "ids": ids, "sequence": str(seq.seq) })
                continue
        rich.print(f"[bold red]{'Error':>12}[/] Sequence has no public database identifier")
    peptides.append(result)

args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w") as dst:
    json.dump(peptides, dst, sort_keys=True, indent=4)

