import argparse
import urllib.request
import os
import io
import json
import tarfile

import pandas
import rich.progress
import taxopy


parser = argparse.ArgumentParser()
parser.add_argument("--blocklist")
parser.add_argument("--taxonomy", "-T", required=True)
parser.add_argument("--mibig-version", default="3.1")
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()

url = f"https://dl.secondarymetabolites.org/mibig/mibig_json_{args.mibig_version}.tar.gz"

with rich.progress.Progress() as progress:
    # load blocklist if any
    if args.blocklist is not None:
        table = pandas.read_table(args.blocklist)
        blocklist = set(table.bgc_id.unique())
    else:
        blocklist = set()
    # download MIBIG 3 records
    mibig = {}
    with urllib.request.urlopen(url) as response:
        total = int(response.headers["Content-Length"])
        with progress.wrap_file(response, total=total, description=f"[bold blue]{'Downloading':>12}[/]") as f:
            with tarfile.open(fileobj=f, mode="r|gz") as tar:
                for entry in iter(tar.next, None):
                    if entry.name.endswith(".json"):
                        with tar.extractfile(entry) as f:
                            data = json.load(f) 
                            if "cluster" in data:
                                data = data["cluster"]
                            accession = data['accession'] if 'accession' in data else data['mibig_accession']
                            if accession not in blocklist:
                                mibig[accession] = data
                            
# load taxonomy database
rich.print(f"[bold blue]{'Loading':>12}[/] taxonomy database from {args.taxonomy!r}")
taxdb = taxopy.TaxDb(taxdb_dir=args.taxonomy, keep_files=True)

# build taxonomy
rows = []
for entry in mibig.values():
    if "ncbi_tax_id" in entry:
        taxid = int(entry["ncbi_tax_id"])
    else:
        taxid = int(entry["taxonomy"]["ncbiTaxId"])
    accession = entry['accession'] if 'accession' in entry else entry['mibig_accession']
    taxon = taxopy.Taxon(taxid, taxdb)
    rows.append({
        "bgc_id": accession,
        "compound": next((
            compound["compound"] if "compound" in compound else compound["name"]
            for compound in entry["compounds"]
        ), ""),
        "tax_id": taxid,
        **taxon.rank_name_dictionary
    })
table = pandas.DataFrame(rows)

# save taxonomy
rich.print(f"[bold green]{'Saving':>12}[/] results to {args.output!r}")
os.makedirs(os.path.dirname(args.output), exist_ok=True)
table.to_csv(args.output, index=False, sep="\t")
