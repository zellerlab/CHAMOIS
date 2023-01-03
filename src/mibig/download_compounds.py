import argparse
import tarfile
import collections
import json
import gzip
import urllib.request
import time
import os

import joblib
import rich.progress
import pubchempy
import pandas
import rdkit.Chem
from rdkit import RDLogger

# disable logging
RDLogger.DisableLog('rdApp.warning')

# get paths from command line
parser = argparse.ArgumentParser()
parser.add_argument("--mibig-version", default="3.1")
parser.add_argument("--atlas", required=True)
parser.add_argument("--blocklist")
parser.add_argument("--cache")
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()

# create persistent cache
if args.cache:
    rich.print(f"[bold green]{'Using':>12}[/] joblib cache folder {args.cache!r}")
    os.makedirs(args.cache, exist_ok=True)
memory = joblib.Memory(location=args.cache, verbose=False)

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
        with progress.wrap_file(response, total=total, description=f"[bold blue]{'Downloading':>12}[/] MIBiG") as f:
            with tarfile.open(fileobj=f, mode="r|gz") as tar:
                for entry in iter(tar.next, None):
                    if entry.name.endswith(".json"):
                        with tar.extractfile(entry) as f:
                            record = json.load(f)["cluster"]
                            if record["mibig_accession"] not in blocklist:
                                mibig[record["mibig_accession"]] = record

rich.print(f"[bold green]{'Downloaded':>12}[/] {len(mibig)} BGCs")

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
                "database_id": ["pubchem:30891"],
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
            for x in "CDE"
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
    elif bgc_id == "BGC0000972":
        entry["compounds"] = [
            {"compound": "colibactin", "database_id": ["pubchem:138805674"]}
        ]
    elif bgc_id == "BGC0001598":
        entry["compounds"] = [
            {"compound": f"foxicin {x}"}
            for x in "ABCD"
        ]
    elif bgc_id == "BGC0001934":
        entry["compounds"] = [
            {"compound": f"armeniaspirol {x}"}
            for x in "ABC"
        ]
    elif bgc_id == "BGC0001961" or bgc_id == "BGC0002288":
        entry["compounds"] = [
            {"compound": f"ashimide {x}"}
            for x in "AB"
        ]
    elif bgc_id == "BGC0001963":
        entry["compounds"] = [
            {"compound": f"catenulisporolide {x}"}
            for x in "ABCD"
        ]
    elif bgc_id == "BGC0001365":
        entry["compounds"] = [
            {
                "compound": "dithioclapurine",
                "chem_struct": r"CC(C)=CCOC1=CC=C(C[C@]23NC(=O)C(NC2=O)SS3)C=C1"
            },
            {
                "compound": "trithioclapurine"
                "chem_struct": r"CC(C)=CCOC1=CC=C(C[C@]23NC(=O)C(NC2=O)SSS3)C=C1"
            },
            {
                "compound": "tetrathioclapurine"
                "chem_struct": r"CC(C)=CCOC1=CC=C(C[C@]23NC(=O)C(NC2=O)SSSS3)C=C1"
            },
        ]
    elif bgc_id == "BGC0002008":
        entry["compounds"] = [
            {
                "compound": "arylpolyene 1",
                "chem_struct": r"C1(C=C(C)C(O)C(C)C=1)/C=C/C=C/C=C/C=C/C=C/C=C/C(OC)=O",  # drawn manually from the paper structure
            },
            { "compound": "arylpolyene 2" },
            { "compound": "arylpolyene 3" },
            { "compound": "arylpolyene 4" },
        ]
    elif bgc_id == "BGC0000380":
        entry["compounds"] = [
            {"compound": f"leupyrrin {x}"}
            for x in ("A1", "A2", "B1", "B2", "C", "D")
        ]
    # only keep the final compounds (sesterfisherol and sesterfisheric acid)
    # and not the intermediates which have a very different topology
    elif bgc_id == "BGC0002162":
        entry["compounds"] = entry["compounds"][:2]
    # the antimycin formula in MIBiG is wrong
    elif bgc_id == "BGC0001455":
        entry["compounds"] = [
            { "compound": f"antimycin A{x+1}{y}" }
            for x in range(4)
            for y in ("a", "b")
        ]
    # the tiancimycin formula in MIBiG is wrong (paper and PubChem agree)
    elif bgc_id == "BGC0001378":
        entry["compounds"] = [
            {
                "compound": "tiancimycin A",
                "database_id": ["pubchem:121477750"],
            }
        ]
    # the SMILES in MIBiG is wrong, but the PubChem xref is correct
    elif bgc_id == "BGC0000657":
        del entry["compounds"][0]["chem_struct"]
    # BGC0000282 produces several alkylresorcinol and alkylcoprinin derivatives
    # (see https://www.jbc.org/article/S0021-9258(20)71631-5/fulltext#gr5);
    # compound 15 which has the highest peak in the heterologous expression profile
    # is recorded below (in pseudo-IUPAC nomenclature, since it is never named)
    elif bgc_id == "BGC0000282":
        entry["compounds"] = [
            {
                "compound": "2-Methoxy-5-methyl-6-isopentadecyl-1,4-benzoquinone",
                "chem_struct": r"COC1=CC(=O)C(C)=C(CCCCCCCCCCCCC(C)C)C1=O",
            }
        ]
    # The IUPAC name of the compound in BGC0002734 is incorrect
    elif bgc_id == "BGC0002734":
        entry["compounds"] = [
            {
                "compound": "4,6-dihydroxy-2,3-dimethylbenzaldehyde",
                "chem_struct": r"CC1=C(C(=C(C=C1O)O)C=O)C",
                "database_id": ["pubchem:254155"]
            }
        ]
    # The compounds for BGC0001917 have not been expanded in MIBiG: in total,
    # there are 14 compounds described in the paper. Minimal cluster
    # (`stmA` to `stmI`) is shown to produce streptoaminal 9i and 9n
    # as well as three 5-alkyltetrahydroquinolines through heterologous
    # expression.
    elif bgc_id == "BGC00001917":
        entry["compounds"] = [
            {
                "compound": "1,2,3,4-tetrahydro-5-isononylquinoline",
                "chem_struct": r"CC(C)CCCCCCC1=CC=CC2=C1CCCN2",
                "database_id": ["pubchem:73053139"],
            },
            {
                "compound": "1,2,3,4-tetrahydro-5-nonylquinoline",
                "chem_struct": r"CCCCCCCCCC1=CC=CC2=C1CCCN2",
                "database_id": ["pubchem:106778340"],
            },
            {
                "compound": "1,2,3,4-tetrahydro-5-decylquinoline",
                "chem_struct": r"CCCCCCCCCCC1=CC=CC2=C1CCCN2",
                "database_id": ["pubchem:106778710"],
            },
            {
                "compound": "streptoaminal-9i",
                "chem_struct": r"CC(C)CCCCCC[C@@H]1C[C@@H](O)C[C@]2(CCCCN2)O1",
                "database_id": ["pubchem:139591491"]
            },
            {
                "compound": "streptoaminal-9n",
                "chem_struct": r"CCCCCCCCC[C@@H]1C[C@@H](O)C[C@]2(CCCCN2)O1",
                "database_id": ["pubchem:132571090"]
            }
        ]
    # Add manually drawn epoxide compounds of BGC0001202
    elif bgc_id == "BGC00001202":
        entry["compounds"] = [
            {
                "compound": "landepoxcin A"
                "chem_struct": r"[H]CC(C)CC(=O)NC([C@@H](C)O)C(=O)N[C@@H](CC(C)=C)C(=O)C1([H])CO1",
            },
            {
                "compound": "landepoxcin B",
                "chem_struct": r"[H]C1(CO1)C(=O)[C@H](CC(C)=C)NC(=O)C(NC(=O)CC(C)CC)[C@@H](C)O",
            }
        ]
    # The correct product of BGC0001829 is TMC-86A, as shown in the
    # paper abstract (see doi:10.1021/jacs.6b01619)
    elif bgc_id == "BGC0001829":
        entry["compounds"] = [
            {
                "compound": "TMC-86A",
                "database_id": ["pubchem:9798121"],
            }
        ]
    # Add manually drawn NRP compounds of BGC0002386 according to
    # the paper (see doi:10.12211/2096-8280.2021-024, Fig. 5)
    elif bgc_id == "BGC0002386":
        entry["compounds"] = [
            {
                "compound": "cyclo(N-methyl-(L)-Leu-(L)-Val)",
                "chem_struct": r"CC(C)C[C@@H]1N(C)C(=O)[C@@H](NC1=O)C(C)C",
            },
            {
                "compound": "cyclo(N-methyl-(L)-Leu-(L)-Leu)",
                "chem_struct": r"CC(C)C[C@@H]1NC(=O)[C@H](CC(C)C)N(C)C1=O",
            },
            {
                "compound": "cyclo(N-methyl-(L)-Leu-(L)-Ile)",
                "chem_struct": r"CC[C@H](C)[C@@H]1NC(=O)[C@H](CC(C)C)N(C)C1=O",
            },
        ]
    # Manually draw the ADEP1 formula using paper reference
    # (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6805094/)
    elif bgc_id == "BGC0001967":
        entry["compounds"] = [
            {
                "compound": "acyldepsipeptide 1",
                "chem_struct": r"C\C=C\C=C\C=C\C(=O)N[C@@H](CC1=CC=CC=C1)C(=O)N[C@H]1COC(=O)[C@@H]2C[C@@H](C)CN2C(=O)[C@H](C)NC(=O)[C@H](C)N(C)C(=O)[C@@H]2CCCN2C1=O",
            }
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
            compound["chem_struct"] = r"Nc1nc2[nH]cc(C#N)c2c(=O)[nH]1"
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
        # add formula of foxicin A (drawn according to paper)
        if compound["compound"] == "foxicin A":
            compound["chem_struct"] = r"C[C@@H](C=C(C)C)\C=C(/C)C(=O)NC1=CC(=O)C(O)=C(NC(C)=O)C1=O"
        # add formula of petrichorin B (drawn according to paper, primary formula matches)
        elif compound["compound"] == "petrichorin B":
            compound["chem_struct"] = r"[H][C@@]12C[C@@]3(O)C4=C(N[C@@]3([H])N1C(=O)[C@H](NC(=O)[C@H](COC)NC(=O)[C@@]1([H])C[C@H](O)CNN1C(=O)[C@@H]1CCCNN1C(=O)[C@H](NC2=O)[C@@H](C)CC)C(C)O)C=C(Cl)C(=C4)C1=CC2=C(N[C@@]3([H])N4C(=O)[C@H](NC(=O)[C@H](COC)NC(=O)[C@@]5([H])C[C@H](O)CNN5C(=O)[C@@H]5CCCNN5C(=O)[C@H](NC(=O)[C@]4([H])C[C@@]23O)[C@@H](C)CC)[C@H](C)O)C=C1Cl"
        # add formula of octapeptin C4 (using octapeptin C8 from PubChem as a base,
        # only the lipid group needs to be changed)
        elif compound["compound"] == "octapeptin C4":
            compound["chem_struct"] = r"CCCCCCCC(O)CC(=O)NC(CCN)C(\O)=N\C1CC\N=C(O)\C(CC(C)C)\N=C(O)\C(CCN)\N=C(O)\C(CCN)\N=C(O)/C(CC(C)C)\N=C(O)/C(CC2=CC=CC=C2)\N=C(O)\C(CCN)\N=C1/O"
        # add formula of BGC0002656 compounds
        elif compound["compound"] == "oryzanaphthopyran A":
            compound["chem_struct"] = r"OC1=C(NC(=O)C2=CC3=C(Cl)C(=O)C4=C(O)C=CC(Cl)=C4C3(O)CO2)C(=O)CC1"
        elif compound["compound"] == "oryzanaphthopyran B":
            compound["chem_struct"] = r"OC1=C(NC(=O)C2=CC3=C(Cl)C(=O)C4=C(O)C=CC=C4C3(O)CO2)C(=O)CC1"
        elif compound["compound"] == "oryzanaphthopyran C":
            compound["chem_struct"] = r"OC1=C(NC(=O)C2=CC3=CC(=O)C4=C(O)C=CC=C4C3(O)CO2)C(=O)CC1"
        elif compound["compound"] == "oryzanthrone A":
            compound["chem_struct"] = r"CC1=C(C(O)=O)C(O)=CC2=C1C(=O)C1=C(O)C=CC=C1C2(C)O"
        elif compound["compound"] == "oryzanthrone B":
            compound["chem_struct"] = r"CC1=CC(O)=CC2=C1C(=O)C1=C(O)C=CC=C1C2(C)O"
        elif compound["compound"] == "chlororyzanthrone A":
            compound["chem_struct"] = r"CC1=CC(O)=C(Cl)C2=C1C(=O)C1=C(O)C=CC=C1[C@H]2O"
        elif compound["compound"] == "chlororyzanthrone B":
            compound["chem_struct"] = r"C[C@@H]1C2=CC=CC(O)=C2C(=O)C2=C1C(Cl)=C(O)C=C2C"
        # add formula of compounds isolated in PMID:25872030
        elif compound["compound"] == "hydroxysporine":
            compound["chem_struct"] = r"OC1=CC2=C(NC3=C2C2=C(C(=O)NC2)C2=C3NC3=C2C=C(O)C=C3)C=C1"
        elif compound["compound"] == "reductasporine":
            compound["chem_struct"] = r"C[N]1(C)CC2=C(C1)C1=C(NC3=C1C=CC=C3)C1=C2C2=C(N1)C=CC=C2"

# --- Load NPAtlas -----------------------------------------------------------
with rich.progress.open(args.atlas, "rb", description=f"[bold blue]{'Loading':>12}[/] NPAtlas") as handle:
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
        rich.print(f"[bold blue]{'Replacing':>12}[/] compounds of [purple]{bgc_id}[/] ({names} with {npnames})")
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
            if len(npaid) > 3 and npaid not in np_atlas:
                npaid_fixed = "NPA{:06}".format(int(npaid[3:]))
                if npaid_fixed in np_atlas:
                    rich.print(f"[bold blue]{'Replacing':>12}[/] broken NPAtlas cross-reference ({npaid!r}) with correct one ({npaid_fixed!r})")
                    xref_index = compound["database_id"].index(npatlas_xref)
                    compound["database_id"][xref_index] = f"npatlas:{npaid_fixed}"
                    compound["chem_struct"] = np_atlas[npaid_fixed]["smiles"]
                else:
                    rich.print(f"[bold blue]{'Removing':>12}[/] broken NPAtlas cross-reference ({npaid!r}) compound {compound['compound']} of [purple]{bgc_id}[/]")
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
                inchikey = rdkit.Chem.inchi.MolToInchiKey(rdkit.Chem.MolFromSmiles(compound['chem_struct'].strip()))
                if inchikey in np_atlas_inchikeys:
                    npaid = np_atlas_inchikeys[inchikey]["npaid"]
                    compound.setdefault("database_id", []).append(f"npatlas:{npaid}")
                    rich.print(f"[bold green]{'Added':>12}[/] cross-reference to NPAtlas compound [bold cyan]{npaid}[/] to {compound['compound']!r} product of [purple]{bgc_id}[/]")
            else:
                compound_name = compound["compound"].casefold()
                if compound_name in np_atlas_names:
                    entry = np_atlas_names[compound_name]
                    compound.setdefault("database_id", []).append(f"npatlas:{entry['npaid']}")
                    compound["chem_struct"] = entry["smiles"]
                    rich.print(f"[bold green]{'Mapped':>12}[/] {compound['compound']!r} product of [purple]{bgc_id}[/] to NPAtlas compound [bold cyan]{entry['npaid']}[/]")
                else:
                    rich.print(f"[bold red]{'Failed':>12}[/] to map {compound['compound']!r} product of [purple]{bgc_id}[/] to NPAtlas")

# --- Try to map unannotated compounds to PubChem ----------------------------

# cache PubChem queries
@memory.cache
def get_cids(name):
    return pubchempy.get_cids(name)

@memory.cache
def get_compounds(cids):
    return pubchempy.get_compounds(cids)

for entry in rich.progress.track(mibig.values(), description=f"[bold blue]{'Mapping':>12}[/]"):
    for compound in entry["compounds"]:
        if "chem_struct" in compound:
            continue
        if not any(xref.startswith("pubchem") for xref in compound.get("database_id", ())):
            name = compound["compound"]
            cids = get_cids(name)
            if cids:
                c = get_compounds(cids[:1])[0]
                compound["database_id"] = [f"pubchem:{c}"]
                compound["chem_struct"] = c.isomeric_smiles
                rich.print(f"[bold green]{'Mapped':>12}[/] {compound['compound']!r} product of [purple]{entry['mibig_accession']}[/] to PubChem compound {c.cid}")
                time.sleep(1)
            else:
                rich.print(f"[bold red]{'Failed':>12}[/] to map {compound['compound']!r} product of [purple]{entry['mibig_accession']}[/] to PubChem")

# --- Retrieve SMILES for compound with a cross-reference --------------------
for entry in rich.progress.track(mibig.values(), description=f"[bold blue]{'Downloading':>12}[/]"):
    for compound in entry["compounds"]:
        # use built-in structure if any
        if "chem_struct" in compound:
            continue
        # use NPAtlas structure if available
        npatlas_xref = next((xref.split(":")[1] for xref in compound.get("database_id", ()) if xref.startswith("npatlas")), None)
        if npatlas_xref is not None:
            np_atlas_entry = np_atlas[npatlas_xref]
            compound["chem_struct"] = np_atlas_entry["smiles"]
            continue
        # use PubChem structure if available
        pubchem_xref = next((int(xref.split(":")[1]) for xref in compound.get("database_id", ()) if xref.startswith("pubchem")), None)
        if pubchem_xref is not None:
            pubchem_entry = get_compounds([pubchem_xref])[0]
            compound["chem_struct"] = pubchem_entry.isomeric_smiles
            continue
        # failed to get the structure...
        rich.print(f"[bold red]{'Failed':>12}[/] to get structure of {compound['compound']!r} product of [purple]{bgc_id}[/]")

# --- Save compounds ---------------------------------------------------------

os.makedirs(os.path.dirname(args.output), exist_ok=True)
compounds = {}

for bgc_id, bgc in mibig.items():
    compounds[bgc_id] = []
    for bgc_compound in bgc["compounds"]:
        compound = { "compound": bgc_compound["compound"] }
        for key in ("chem_struct", "database_id", "mol_mass", "molecular_formula"):
            if key in bgc_compound:
                compound[key] = bgc_compound[key]
        compounds[bgc_id].append(compound)

with open(args.output, "w") as dst:
    json.dump(compounds, dst, sort_keys=True, indent=4)
