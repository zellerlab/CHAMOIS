import argparse
import tarfile
import collections
import json
import gzip
import urllib.request
import time
import os

import rich.progress
import pubchempy
from openbabel import pybel

# Disable warnings from OpenBabel
pybel.ob.obErrorLog.SetOutputLevel(0)

# get paths from command line
parser = argparse.ArgumentParser()
parser.add_argument("--mibig-version", default="3.1")
parser.add_argument("--atlas", required=True)
parser.add_argument("--blocklist")
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()


# --- Download MIBiG metadata ------------------------------------------------

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
        with progress.wrap_file(response, total=total, description="Downloading...") as f:
            with tarfile.open(fileobj=f, mode="r|gz") as tar:
                for entry in iter(tar.next, None):
                    if entry.name.endswith(".json"):
                        with tar.extractfile(entry) as f:
                            record = json.load(f)["cluster"]
                            if record["mibig_accession"] not in blocklist:
                                mibig[record["mibig_accession"]] = record

print("Downloaded {} BGCs".format(len(mibig)))

# --- Manual mapping of some compounds ---------------------------------------
for bgc_id, entry in mibig.items():
    
    if bgc_id in ("BGC0000243", "BGC0000244"):
        entry["compounds"] = [
            {"compound": name}
            for name in ["nonactin", "monactin", "dinactin", "trinactin", "tetranactin"]
        ]
    elif bgc_id == "BGC0001465":
        entry["compounds"] = [
            {
                "compound": "bromophene",
                "database_id": [f"pubchem:30891"],
            },
            {
                "compound": "bistribromopyrrole", 
                "database_id": [f"pubchem:23426790"],
            },
            {
                "compound": "pentabromopseudilin",
                "database_id": [f"pubchem:324093"],
            }
            
        ]
        continue
    elif bgc_id == "BGC0000248":
        entry["compounds"] = [
            {
                "compound": "alpha-naphthocyclinone",
                "database_id": [f"pubchem:102267544"]
            }
        ]
        continue
    elif bgc_id == "BGC0000986":
        entry["compounds"] = [
            {"compound": f"DKxanthene {x}"}
            for x in (504, 518, 530, 544, 556)
        ]
        continue
    elif bgc_id == "BGC0001413":
        entry["compounds"] = [
            {"compound": f"cystobactamid 919-{x}"}
            for x in (1, 2, 3)
        ]
        continue
    elif bgc_id == "BGC0001983":
        entry["compounds"] = [
            {"compound": f"triacsin C"},
        ]
        continue
    elif bgc_id == "BGC0002019":
        entry["compounds"] = [
            {"compound": f"tiancilactone {x}"}
            for x in "ABCDEFGH"
        ]
    elif bgc_id == "BGC0001620":
        entry["compounds"] = [
            {"compound": f"ilamycin {x}"}
            for x in ("B1", "B2", "C1", "C2", "D", "E1")
        ]
    elif bgc_id == "BGC0001625":
        entry["compounds"] = [
            {"compound": f"isofuranonaphthoquinone {x}"}
            for x in "ABCDEFG"
        ]
    elif bgc_id == "BGC0001730":
        entry["compounds"] = [
            {"compound": f"paramagnetoquinone {x}"}
            for x in "ABC"
        ]
    elif bgc_id == "BGC0001596":
        entry["compounds"] = [
            {"compound": f"fluostatin {x}"}
            for x in "MNOPQ"
        ]
    elif bgc_id == "BGC0001666":
        entry["compounds"] = [
            {"compound": f"microansamycin {x}"}
            for x in "ABCDEFGHI"
        ]
    elif bgc_id == "BGC0001590":
        entry["compounds"] = [
            {"compound": f"formicamycin {x}"}
            for x in "ABCDEFGHIJKLM"
        ]
    elif bgc_id == "BGC0001755":
        entry["compounds"] = [
            {"compound": f"reedsmycin {x}"}
            for x in "ABCDE"
        ]
    elif bgc_id == "BGC0001726":
        entry["compounds"] = [
            {"compound": f"pactamide {x}"}
            for x in "CDEF"
        ]
    elif bgc_id == "BGC0000341":
        entry["compounds"] = [
            {"compound": f"enduracidin {x}"}
            for x in "AB"
        ]
    elif bgc_id == "BGC0001700":
        entry["compounds"] = [
            {"compound": f"niphimycin {x}"}
            for x in ""
        ]
    elif bgc_id == "BGC0000289":
        entry["compounds"] = [
            {"compound": f"A-40926 {x}"}
            for x in ("A", "B", "PA", "PB")
        ]
    elif bgc_id == "BGC0000418":
        entry["compounds"][0] = {"compound": "ristocetin A"}
    elif bgc_id == "BGC0002396":
        entry["compounds"] = [
            {"compound": f"tanshinone {x}"}
            for x in ("I", "IIA")
        ]

    for compound in entry["compounds"]:
        # β-D-galactosylvalidoxylamine-A is actually validamycin
        if compound["compound"] == "β-D-galactosylvalidoxylamine-A":
            compound["compound"] = "validamycin A"
        # fusarin BGCs actually produce fusarin C
        elif compound["compound"] == "fusarin":
            compound["compound"] = "fusarin C"
        elif compound["compound"] == "PreQ0 Base":
            compound["compound"] = "7-cyano-7-deazaguanine"
            compound["chem_struct"] = "Nc1nc2[nH]cc(C#N)c2c(=O)[nH]1"
        # aureusimine BGCs actually produce aureusimine A, B and C
        elif compound["compound"] == "aureusimine":
            entry["compounds"] = [{"compound": f"aureusimine {x}"} for x in "ABC"]
            break
        # griseusin BGCs actually produce griseusin A and B
        elif compound["compound"] == "griseusin":
            entry["compounds"] = [{"compound": f"griseusin {x}"} for x in "AB"]
            break
        # pelgipeptin BGCs actually produce pelgipeptin A and B
        elif compound["compound"] == "pelgipeptin":
            entry["compounds"] = [{"compound": f"pelgipeptin {x}"} for x in "AB"]
            break
        # piricyclamide BGCs actually produce piricyclamide 7005E1-4
        elif compound["compound"] == "piricyclamide":
            entry["compounds"] = [{"compound": f"piricyclamide 7005E{x}"} for x in "1234"]
            break
        #
        elif compound["compound"] == "burnettramic acid":
            entry["compounds"] = [{"compound": f"burnettramic acid {x}"} for x in "AB"]
            break
        #
        elif compound["compound"] == "paenilarvins":
            entry["compounds"] = [{"compound": f"paenilarvin {x}"} for x in "ABC"]
            break
        #
        elif compound["compound"] == "hassallidin C":
            entry["compounds"] = [{"compound": f"hassallidin {x}"} for x in "ABCD"]
            break
        # 
        elif compound["compound"] == "odilorhabdins":
            entry["compounds"] = [{"compound": f"odilorhabdin NOSO-95{x}"} for x in "ABC"]
            break
        #
        elif compound["compound"] == "pactamides":
            entry["compounds"] = [{"compound": f"pactamide {x}"} for x in "AB"]
            break
        elif compound["compound"] == "grixazone":
            compound["compound"] = "grixazone A"
        elif compound["compound"] == "luminmide":
            entry["compounds"] = [{"compound": f"luminmide {x}"} for x in "AB"]
            break
        elif compound["compound"] == "aspercryptins":
            compound["compound"] = "aspercryptin A1"
        elif "bartolosides" in compound["compound"]:
            compound["compound"] = compound["compound"].replace("bartolosides", "bartoloside")
        elif compound["compound"] == "splenocin":
            compound["compound"] = "splenocin C" # see PMID:25763681, "in our hands, we only observe SPN-C in the fermentation of CNQ431"
        elif compound["compound"] == "fogacin A":
            compound["compound"] = "fogacin"
        elif compound["compound"] == "lacunalides":
            entry["compounds"] = [{"compound": f"lacunalide {x}"} for x in "AB"]
        # fix names of bartolosides
        if compound["compound"] == "bartoloside 2":
            compound["compound"] = "bartoloside B"
        elif compound["compound"] == "bartoloside 3":
            compound["compound"] = "bartoloside C"
        elif compound["compound"] == "bartoloside 4":
            compound["compound"] = "bartoloside D"
        # fix names of fortimicin
        if compound["compound"] == "fortimicin":
            compound["compound"] = "fortimicin A"

# --- Load NPAtlas -----------------------------------------------------------
with rich.progress.open(args.atlas, "rb", description="Loading NPAtlas...") as handle:
    data = json.load(gzip.open(handle))

np_atlas = {
    entry["npaid"]: entry
    for entry in rich.progress.track(data, description="Indexing NPAtlas...")    
}
del data

# --- Replace compounds from MIBiG that have a NPAtlas annotation ------------
mibig_entries = collections.defaultdict(list)
for npaid, entry in rich.progress.track(np_atlas.items(), description="Patching MIBiG..."):
    mibig_xrefs = [xref["external_db_code"] for xref in entry["external_ids"] if xref["external_db_name"] == "mibig"]
    for xref in mibig_xrefs:
        mibig_entries[xref].append(entry)

for bgc_id, bgc in mibig.items():
    if bgc_id in mibig_entries and len(bgc["compounds"]) == 1 and "chem_struct" not in bgc["compounds"][0]:
        names = ", ".join(repr(compound["compound"]) for compound in bgc['compounds'])
        npnames = ", ".join(repr(entry['original_name']) for entry in mibig_entries[bgc_id])
        rich.print(f"Replacing compounds of {bgc_id} ({names} with {npnames})")
        bgc['compounds'] = [
            {
                "compound": entry["original_name"], 
                "chem_struct": entry["smiles"], 
                "database_id": [f"npatlas:{entry['npaid']}"],
            }
            for entry in mibig_entries[bgc_id]
        ]

# --- Fix broken NPAtlas cross-references ------------------------------------

for bgc_id, bgc in mibig.items():
    for compound in bgc["compounds"]:
        npatlas_xref = next(
            (xref for xref in compound.get("database_id", ()) if xref.startswith("npatlas:")),
            None
        )
        if npatlas_xref is not None:
            npaid = npatlas_xref.split(":")[1]
            if npaid not in np_atlas:
                npaid_fixed = "NPA{:06}".format(int(npaid[3:]))
                if npaid_fixed in np_atlas:
                    rich.print(f"Replacing broken NPAtlas cross-reference ({npaid!r}) with correct one ({npaid_fixed!r})")
                    xref_index = compound["database_id"].index(npatlas_xref)
                    compound["database_id"][xref_index] = f"npatlas:{npaid_fixed}"
                else:
                    rich.print(f"Removing broken NPAtlas cross-reference ({npaid!r}) compound {compound['compound']} of {bgc_id}")
                    compound["database_id"].remove(npatlas_xref)


# --- Try to map unannotated compounds to NPAtlas ----------------------------
np_atlas_inchikeys = {
    entry["inchikey"]: entry
    for entry in np_atlas.values()
}
np_atlas_names = {
    entry["original_name"].casefold(): entry
    for entry in np_atlas.values()
}
for bgc_id, bgc in mibig.items():
    for compound in bgc["compounds"]:
        if not any(xref.startswith("npatlas") for xref in compound.get("database_id", ())):
            if "chem_struct" in compound:
                inchikey = pybel.readstring("smi", compound['chem_struct'].strip()).write("inchikey").strip()
                if inchikey in np_atlas_inchikeys:
                    npaid = np_atlas_inchikeys[inchikey]["npaid"]
                    compound.setdefault("database_id", []).append(f"{npaid}")
                    rich.print(f"Added cross-reference to NPAtlas compound {npaid} to {compound['compound']!r} product of {bgc_id}")
                    continue
            else:
                compound_name = compound["compound"].casefold()
                if compound_name in np_atlas_names:
                    entry = np_atlas_names[compound_name]
                    compound.setdefault("database_id", []).append(f"npatlas:{entry['npaid']}")
                    compound["chem_struct"] = entry["smiles"]
                    rich.print(f"Mapped {compound['compound']!r} product of {bgc_id} to NPAtlas compound {entry['npaid']}")
                else:
                    rich.print(f"Failed to map {compound['compound']!r} product of {bgc_id}") 


# --- Extract remaining missing structures from PubChem ----------------------
for entry in rich.progress.track(mibig.values()):
    for compound in entry["compounds"]:
        if "chem_struct" not in compound:   
            name = compound["compound"]
            cids = pubchempy.get_cids(name)
            if cids:
                c = pubchempy.get_compounds(cids[:1])[0]
                compound["database_id"] = [f"pubchem:{c}"]
                compound["chem_struct"] = c.isomeric_smiles
                rich.print(f"Mapped {compound['compound']!r} of {entry['mibig_accession']} to PubChem compound {c.cid}")
                time.sleep(1)
            else:
                rich.print(f"Failed to map {compound['compound']!r} product of {entry['mibig_accession']}") 

# --- Save compounds ---------------------------------------------------------

os.makedirs(os.path.dirname(args.output), exist_ok=True)
compounds = {
    cluster["mibig_accession"]: cluster["compounds"]
    for cluster in mibig.values()
}

with open(args.output, "w") as dst:
    json.dump(compounds, dst, sort_keys=True, indent=4)      
