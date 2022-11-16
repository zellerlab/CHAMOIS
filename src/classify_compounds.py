import argparse
import collections
import json
import gzip
import urllib.request
import time
from urllib.error import HTTPError
from typing import Dict, List

import pronto
import rich.progress
import scipy.sparse
from openbabel import pybel

# Disable warnings from OpenBabel
pybel.ob.obErrorLog.SetOutputLevel(0)

# get paths from command line
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
parser.add_argument("--atlas", required=True)
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()


# --- Load MIBiG -------------------------------------------------------------

with rich.progress.open(args.input, "rb", description="Loading MIBiG...") as handle:
    mibig = json.load(handle)


# --- Load NPAtlas -----------------------------------------------------------

with rich.progress.open(args.atlas, "rb", description="Loading NPAtlas...") as handle:
    data = json.load(gzip.open(handle))

np_atlas = {
    entry["npaid"]: entry
    for entry in rich.progress.track(data, description="Indexing NPAtlas...")    
}
inchikey_index = {
    entry["inchikey"]: entry
    for entry in rich.progress.track(data, description="Indexing NPAtlas...")    
}


# --- Get ClassyFire annotations for all compounds ----------------------------

for bgc_id, bgc in rich.progress.track(mibig.items(), description="Classifying compounds..."):
    for compound in bgc["compounds"]:
        # attempt to get classyfire annotation from NPAtlas
        npaid = next((dbid.split(":")[1] for dbid in compound.get("database_id", ()) if dbid.startswith("npatlas")), None)
        if npaid is not None:
            if npaid not in np_atlas:
                rich.print(f"Skipping invalid NPAtlas xref ({npaid}) of compound {compound['compound']!r} of {bgc_id}")
            elif np_atlas[npaid]["classyfire"] is not None:
                rich.print(f"Using NPAtlas classification ({npaid}) for compound {compound['compound']!r} of {bgc_id}")
                compound["classyfire"] = np_atlas[npaid]["classyfire"]
                assert compound["classyfire"] is not None
                continue
        # ignore compounds without structure (should have gotten one)
        if "chem_struct" not in compound:
            rich.print(f"Skipping {compound['compound']!r} compound of {bgc_id} with no structure")
            continue
        # try to use classyfire by inchi
        inchikey = pybel.readstring("smi", compound['chem_struct'].strip()).write("inchikey").strip()
        if inchikey in inchikey_index:
            npaid = inchikey_index[inchikey]["npaid"]
            compound.setdefault("database_id", []).append(f"npatlas:{npaid}")
            if np_atlas[npaid]["classyfire"] is not None:
                rich.print(f"Using NPAtlas classification ({npaid}) for compound {compound['compound']!r} of {bgc_id}")
                compound["classyfire"] = np_atlas[npaid]["classyfire"]
                assert compound["classyfire"] is not None
                continue
        # otherwise use the ClassyFire website API
        try:
            rich.print(f"Querying ClassyFire for compound {compound['compound']!r} of {bgc_id}")
            with urllib.request.urlopen(f"http://classyfire.wishartlab.com/entities/{inchikey}.json") as res:
                if res.code == 200:
                    data = json.load(res)
                    if "class" in data:
                        compound["classyfire"] = data
                        assert compound["classyfire"] is not None
        except HTTPError:
            continue
        finally:
            time.sleep(1) # avoid spamming the server


# --- Save -------------------------------------------------------------------

with open(args.output, "w") as dst:
    json.dump(mibig, dst)      

