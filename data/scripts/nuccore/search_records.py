import argparse
import itertools
import json
import gzip
import csv
import pickle
import os
import time
import math
import random
import tarfile
import urllib.request

import Bio.Entrez
import Bio.SeqIO
import pubchempy
import rich.progress
import rapidfuzz.process
from rich.prompt import Confirm, Prompt
from rich.progress import TextColumn, BarColumn, MofNCompleteColumn, TaskProgressColumn, TimeRemainingColumn

parser = argparse.ArgumentParser()
parser.add_argument("--atlas", required=True)
parser.add_argument("--compounds", required=True)
parser.add_argument("--mibig-version", default="3.1")
parser.add_argument("--email", default="martin.larralde@embl.de")
args = parser.parse_args()

Bio.Entrez.email = args.email

# --- Download MIBiG metadata ------------------------------------------------

url = f"https://dl.secondarymetabolites.org/mibig/mibig_json_{args.mibig_version}.tar.gz"

with rich.progress.Progress() as progress:
    # download MIBIG 3 records
    mibig = {}
    with urllib.request.urlopen(url) as response:
        total = int(response.headers["Content-Length"])
        with progress.wrap_file(response, total=total, description=f"[bold blue]{'Downloading':>12}[/] MIBiG") as f:
            with tarfile.open(fileobj=f, mode="r|gz") as tar:
                for entry in iter(tar.next, None):
                    if entry.name.endswith(".json"):
                        with tar.extractfile(entry) as f:
                            record = json.load(f)["cluster"]
                            mibig[record["mibig_accession"]] = record

rich.print(f"[bold green]{'Downloaded':>12}[/] {len(mibig)} BGCs")

mibig_loci = { 
    bgc["loci"]["accession"].rsplit(".", 1)[0]:bgc_id
    for bgc_id, bgc in mibig.items() 
}

# --- Load NPAtlas -----------------------------------------------------------

with rich.progress.open(args.atlas, "rb", description=f"[bold blue]{'Loading':>12}[/]", transient=True) as handle:
    np_atlas = {
        entry["npaid"]: entry 
        for entry in json.load(gzip.open(handle))
    }
    np_by_name = {
        entry["original_name"]: entry
        for entry in np_atlas.values()
    }
    np_names = list(np_by_name)

# --- Load existing data -----------------------------------------------------

if os.path.exists(args.compounds):
    with open(args.compounds) as f:
        compounds = json.load(f)
else:
    compounds = {}

# --- Query NCBI -------------------------------------------------------------

with rich.progress.Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TaskProgressColumn(),
    TimeRemainingColumn(),
) as progress:
    # get all accessions of records named "biosynthetic gene cluster"
    accessions = set()
    task_id = progress.add_task(total=None, description=f"[bold blue]{'Searching':>12}[/]")
    pos, total = 0, math.inf
    while pos < total:
        # dowload search page
        handle = Bio.Entrez.esearch(
            db="nucleotide",
            # term='"biosynthetic gene cluster" "complete sequence" bacteria[filter]',
            term='"biosynthetic gene cluster"[All Fields] AND bacteria[filter] NOT shotgun[All Fields]',
            retstart=pos,
            idtype="acc",
            sort="date",
            retmax=200,
        )
        record =  Bio.Entrez.read(handle)
        # download info for search results
        try:
            entries = Bio.Entrez.parse(Bio.Entrez.esummary(id=",".join(record["IdList"]), db="nuccore", retmax=200))
            accessions.update(entry['AccessionVersion'] for entry in entries if entry['Caption'] not in mibig_loci)
        except RuntimeError:
            continue
        finally:
            time.sleep(1.0)
        # update progress
        total = int(record['Count'])
        pos += len(record['IdList'])
        progress.update(task_id=task_id, total=total, completed=pos)
    progress.update(task_id=task_id, visible=False)
    progress.remove_task(task_id)


try:
    # batch download from GenBank
    accessions = list(accessions)
    random.shuffle(accessions)
    for accession in accessions:
        # with Bio.Entrez.efetch(db="nucleotide", id=",".join(accessions), rettype="gbwithparts", retmode="text") as handle:
            # reader = Bio.SeqIO.parse(handle, "genbank")
        with Bio.Entrez.efetch(db="nucleotide", id=accession, rettype="gbwithparts", retmode="text") as handle:
            record = Bio.SeqIO.read(handle, "genbank")
        # for record in reader:
        # ignore false-positives
        if any(x in record.description for x in ["pyocin", "t3pks", "metabolite", "PUFA", ]):
            continue
        # find feature for BGCs in the record, ignore multi-BGC record
        bgc_features = [
            (feat.qualifiers["note"][0], feat.location)
            for feat in record.features
            if feat.type == "misc_feature"
            and "biosynthetic gene cluster" in feat.qualifiers.get("note", [""])[0]
        ]
        if len(bgc_features) > 1:
            continue
        bgc_id = f"{record.id}_cluster1"
        # extract compound
        if bgc_id in compounds:
            rich.print(f"[bold green]{'Using':>12}[/] cached compound for [purple]{record.id}[/]: [cyan]{compounds[bgc_id][0]['compound']}[/]")
        else:
            # extract product from title
            rich.print(f"[bold blue]{'Searching':>12}[/] compound name for [purple]{record.id}[/]: {record.description}")
            fields = record.description.split()
            try:
                x = fields.index("biosynthetic")
                compound = fields[x-1]
                if len(compound) <= 2 or compound == "acid":
                    compound = ' '.join(fields[x-2:x])
            except ValueError:
                rich.print(f"[bold red]{'Failed':>12}[/] to find compound for [purple]{bgc_id}[/]")
                continue
            # ask user for a compound if automatic extraction failed
            rich.print(f"[bold blue]{'Using':>12}[/] compound name for [purple]{record.id}[/]: [cyan]{compound}[/]")
            ok = Confirm.ask(f"[bold purple]{'Use':>12}[/] a different compound name?", default=False)
            if ok:
                compound = Prompt.ask(f"[bold purple]{'Give':>12}[/] the right compound name")
            # search compound in NP atlas
            use_npatlas = Confirm.ask(f"[bold purple]{'Search':>12}[/] compound in NPAtlas?", default=True)
            if use_npatlas:
                np_name, score, _ = rapidfuzz.process.extractOne(compound, np_names)
                np_entry = np_by_name[np_name]
                rich.print(f"[bold blue]{'Found':>12}[/] compound for [purple]{bgc_id}[/]: [cyan]{np_name}[/] ([cyan]{np_entry['npaid']}[/])")
                ok = Confirm.ask(f"[bold purple]{'Use':>12}[/] this compound?", default=True)
                if ok:
                    np_entry = np_by_name[np_name]
                    compounds[bgc_id] = [{
                        "compound": np_name,
                        "database_id": [f"npatlas:{np_entry['npaid']}"],
                        "chem_struct": np_entry['smiles'],
                    }]
                    continue
            # search compound in PubChem
            use_pubchem = Confirm.ask(f"[bold purple]{'Search':>12}[/] compound in PubChem?", default=True)
            if use_pubchem:               
                pc_compounds = pubchempy.get_compounds(compound, 'name')
                if pc_compounds:
                    pc_name = pc_compounds[0].synonyms[0]
                    rich.print(f"[bold blue]{'Found':>12}[/] compound for [purple]{bgc_id}[/]: [cyan]{pc_name}[/] ([cyan]PubChem:{pc_compounds[0].cid}[/])")
                    ok = Confirm.ask(f"[bold purple]{'Use':>12}[/] this compound?", default=True)
                    if ok:
                        compounds[bgc_id] = [{
                            "compound": pc_name,
                            "database_id": [f"pubchem:{pc_compounds[0].cid}"],
                            "chem_struct": pc_compounds[0].isomeric_smiles,
                        }]
                        continue

            # skip compound on failure
            rich.print(f"[bold red]{'Failed':>12}[/] to find compound for [purple]{bgc_id}[/]")
except BaseException as err:
    rich.print(f"[bold red]{'Interrupted':>12}[/] {err}")
finally:
    # save compounds to output
    os.makedirs(os.path.dirname(args.compounds), exist_ok=True)
    with open(args.compounds, "w") as dst:
        json.dump(compounds, dst, sort_keys=True, indent=4)
        
    
