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

def convert_mibig4_to_mibig3(entry):
    entry["mibig_accession"] = entry.pop("accession")
    for compound in entry["compounds"]:
        if "name" in compound:
            compound["compound"] = compound.pop("name")
        if "structure" in compound:
            compound["chem_struct"] = compound.pop("structure")
    return entry

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
                            record = json.load(f)
                            cluster = convert_mibig4_to_mibig3(record) if args.mibig_version == "4.0" else record["cluster"]
                            if cluster["status"] == "retired":
                                continue
                            if cluster["mibig_accession"] in blocklist:
                                continue
                            mibig[cluster["mibig_accession"]] = cluster

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
                "compound": "trithioclapurine",
                "chem_struct": r"CC(C)=CCOC1=CC=C(C[C@]23NC(=O)C(NC2=O)SSS3)C=C1"
            },
            {
                "compound": "tetrathioclapurine",
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
                "compound": "landepoxcin A",
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
    # Add the PubChem reference to BGC0001847 compound
    elif bgc_id == "BGC0001847":
        entry["compounds"] = [
            {
                "compound": "1-carbapen-2-em-3-carboxylic acid",
                "chem_struct": "C1C=C(N2[C@H]1CC2=O)C(=O)O",
                "database_id": ["pubchem:441133"]
            }
        ]
    elif bgc_id == "BGC0002636":
        entry["compounds"] = [
            {
                "compound": "N-acetyl-cysteinylated streptophenazine A",
                "chem_struct": r"C1=CC=C(C(=O)OC)C2=NC3=CC=CC([C@@](SC[C@@](C(O)=O)NC(=O)C)[C@@](CCCC(C)C)C(=O)OC)=C3N=C12",
            },
            {
                "compound": "N-acetyl-cysteinylated streptophenazine F",
                "chem_struct": r"C1=CC=C(C(=O)OC)C2=NC3=CC=CC([C@@](SC[C@@](C(O)=O)NC(=O)C)[C@@](CCCCC(C)C)C(=O)OC)=C3N=C12",
            },
        ]
    elif bgc_id == "BGC0002624":
        entry["compounds"] = [
            {
                "compound": "fischerazole A",
                "chem_struct": r"C(/Cl)(\Cl)=C/C[C@@](Cl)CCC[C@@](O)(C=C)CCCC[C@](CC1SC=C(C(=O)NC)N=1)Cl",
            },
            {
                "compound": "fischerazole B",
                "chem_struct": r"C(/Cl)(\Cl)=C/CCCCC[C@@](O)(C=C)CCCC[C@](CC1SC=C(C(=O)NC)N=1)Cl",
            },
            {
                "compound": "fischerazole C",
                "chem_struct": r"C(/Cl)(\Cl)=C/C[C@@](Cl)CCC[C@@]([H])(C=C)CCCC[C@](CC1SC=C(C(=O)NC)N=1)Cl",
            },
        ]
    elif bgc_id == "BGC0002597":
        entry["compounds"] = [
            {
                "compound": "nocuolactylate A",
                "database_id": ["npatlas:NPA033255"],
            },
            {
                "compound": "nocuolactylate B",
                "database_id": ["npatlas:NPA033256"],
            },
            {
                "compound": "nocuolactylate C",
                "database_id": ["npatlas:NPA033257"],
            },
        ]
    elif bgc_id == "BGC0002402":
        entry["compounds"] = [
            {
                "compound": "yossoside I",
                "chem_struct": r"C1[C@](O)[C@](O[C@@]2O[C@](C(=O)O)[C@@](O)[C@](O)[C@]2O)[C@](C(=O)O)(C)C2CC[C@](C)3[C@@](C)4CC[C@](C(O[C@]5[C@](O)[C@@](O)[C@@](O)[C@@](C)O5)=O)5CC[C@@](C)(C)CC5C4=CCC3[C@@]12C",
            },
            {
                "compound": "yossoside II",
                "chem_struct": r"C1[C@](O)[C@](O[C@@]2O[C@](C(=O)O)[C@@](O)[C@](O)[C@]2O)[C@](C(=O)O)(C)C2CC[C@](C)3[C@@](C)4CC[C@](C(O[C@]5[C@](O[C@]6[C@](O)[C@](O)[C@@](O)[C@](C)O6)[C@@](O)[C@@](O)[C@@](C)O5)=O)5CC[C@@](C)(C)CC5C4=CCC3[C@@]12C",
            },
            {
                "compound": "yossoside III",
                "chem_struct": r"C1[C@](O)[C@](O[C@@]2O[C@](C(=O)O)[C@@](O)[C@](O)[C@]2O)[C@](C(=O)O)(C)C2CC[C@](C)3[C@@](C)4CC[C@](C(O[C@]5[C@](O[C@]6[C@](O)[C@](O)[C@@](O[C@@]7O[C@](CO)[C@@](O)[C@](O)[C@]7O)[C@](C)O6)[C@@](O)[C@@](O)[C@@](C)O5)=O)5CC[C@@](C)(C)CC5C4=CCC3[C@@]12C",
            },
            {
                "compound": "yossoside IV",
                "chem_struct": r"C1[C@](O)[C@](O[C@@]2O[C@](C(=O)O)[C@@](O)[C@](O[C@]3[C@](O)[C@@](O)[C@](O)CO3)[C@]2O)[C@](C(=O)O)(C)C2CC[C@](C)3[C@@](C)4CC[C@](C(O[C@]5[C@](O[C@]6[C@](O)[C@](O)[C@@](O[C@@]7O[C@](CO)[C@@](O)[C@](O)[C@]7O)[C@](C)O6)[C@@](O)[C@@](O)[C@@](C)O5)=O)5CC[C@@](C)(C)CC5C4=CCC3[C@@]12C",
            },
            {
                "compound": "yossoside V",
                "chem_struct": r"C1[C@](O)[C@](O[C@@]2O[C@](C(=O)O)[C@@](O)[C@](O[C@]3[C@](O)[C@@](O)[C@](O)CO3)[C@]2O)[C@](C(=O)O)(C)C2CC[C@](C)3[C@@](C)4CC[C@](C(O[C@]5[C@](O[C@]6[C@](O)[C@](O)[C@@](O[C@@]7O[C@](CO)[C@@](O)[C@](O)[C@]7O)[C@](C)O6)[C@@](O)[C@@](OC(=O)C)[C@@](C)O5)=O)5CC[C@@](C)(C)CC5C4=CCC3[C@@]12C",
            },
        ]
    elif bgc_id == "BGC0002336":
        entry["compounds"] = [
            {
                "compound": "gladiochelin A",
                "chem_struct": r"C(O)(=O)CC(CCCC/C=C\CCCCCCC/C=C\C(N/C=C/C(N[C@]1COC(=O)[C@](CO)NC(=O)[C@@](NC(=O)[C@@](C(C)C)NC1=O)CCO)=O)=O)(C(O)=O)OC",
            },
            {
                "compound": "gladiochelin B",
                "chem_struct": r"C(O)(=O)CC(CCCC/C=C\CCCCCCC/C=C\C(N/C=C/C(N[C@]1COC(=O)[C@](CO)NC(=O)[C@@](NC(=O)[C@@]([C@@](CC)C)NC1=O)CCO)=O)=O)(C(O)=O)OC",
            }
        ]
    # Use the non-salt version of lasalocid
    elif bgc_id == "BGC0000086" or bgc_id == "BGC0000087":
        entry["compounds"] = [
            {
                "compound": "lasalocid",
                "database_id": ["pubchem:5360807"]
            }
        ]
    # Use the non-salt version of tetronomycin
    elif bgc_id == "BGC0000164":
        entry["compounds"] = [
            {
                "compound": "tetronomycin",
                "database_id": ["pubchem:54717181"],
            }
        ]
    # Fix annotated compounds of tartrolon BGC
    # (see https://pubmed.ncbi.nlm.nih.gov/23288898/)
    elif bgc_id == "BGC0000185":
        entry["compounds"] = [
            {
                "compound": "tartrolon D",
                "database_id": ["pubchem:44614261"],
            },
            {
                "compound": "tartrolon E",
            }
        ]
    # kinamycin BGC produced kinamycin D
    # (see https://pubmed.ncbi.nlm.nih.gov/9531987/)
    elif bgc_id == "BGC0000236":
        entry["compounds"] = [
            {
                "compound": "kinamycin D",
                "database_id": ["pubchem:135440049"],
            }
        ]
    # BGC0000404 produces penicillin N and isopenicillin N
    # (see https://pubmed.ncbi.nlm.nih.gov/16713314/)
    elif bgc_id == "BGC0000404":
        entry["compounds"] = [
            {
                "compound": "isopenicillin N",
                "database_id": ["pubchem:440723"],
            },
            {
                "compound": "penicillin N",
                "database_id": ["pubchem:71724"],
            },
        ]
    # BGC0000548 produces salivaricin A, which is very broken on PubChem,
    # so this is the proper compound SMILES drawn from precursor petpide with PTM
    # (see https://www.sciencedirect.com/science/article/pii/S0966842X20300585, Fig.4)
    elif bgc_id == "BGC0000548":
        entry["compounds"] = [
            {
                "compound": "salivaricin A",
                "chem_struct": r"N[C@@H](CCCCN)C(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)N[C@@H](CO)C(=O)NCC(=O)N[C@@H](Cc1cNc2c1cccc2)C(=O)N[C@@H]([C@@H](C)CC)C(=O)N[C@@H](C)C(=O)N[C@@H]([C@@H]3C)C(=O)N[C@@H]([C@@H](C)CC)C(=O)N[C@@H]([C@@H]4C)C(=O)N[C@@H](CC(O)=O)C(=O)N[C@@H](CC(O)=O)C(=O)N[C@@H](CS3)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CC(=O)N)C(=O)N[C@@H](C5)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CS4)C(=O)N[C@@H](CS5)C(=O)-O",
            }
        ]
    # Use iron-free ferrichrome formula produced by BGC0000901
    # (see https://pubmed.ncbi.nlm.nih.gov/18680426/)
    elif bgc_id == "BGC0000901":
        entry["compounds"] = [
            {
                "compound": "ferrichrome",
                "database_id": ["pubchem:23451539"]
            }
        ]
    # Use iron-free heme-d(1) formula (porphyrindione) produced by BGC0000906
    elif bgc_id == "BGC0000906":
        entry["compounds"] = [
            {
                "compound": "porphyrindione",
                "database_id": ["pubchem:6438546"]
            }
        ]
    # Use the non-complexed molybdenum cofactor molecule (molydopterin)
    # for BGC0000916 and BGC0000917
    elif bgc_id == "BGC0000916" or bgc_id == "BGC0000917":
        entry["compounds"] = [
            {
                "compound": "molydopterin",
                "database_id": ["pubchem:135398581"]
            }
        ]
    # Use the right mycobactin compound produced by BGC0001021
    elif bgc_id == "BGC0001021":
        entry["compounds"] = [
            {
                "compound": "mycobactin",
                "database_id": ["pubchem:3083702", "npatlas:NPA024789"],
                "chem_struct": r"C1CCN(C(=O)C(C1)NC(=O)CCOC(=O)C(CCCCCN(C=O)O)NC(=O)C2COC(=N2)C3=CC=CC=C3O)O",
            }
        ]
    # Use the iron-free, neutral charge compound for BGC0001249
    elif bgc_id == "BGC0001249":
        entry["compounds"] = [
            {
                "compound": "dimethyl coprogen",
                "database_id": ["pubchem:24954742"]
            }
        ]
    # Use the nickel-free compound for BGC0001554
    elif bgc_id == "BGC0001554":
        entry["compounds"] = [
            {
                "compound": "coenzyme F430",
                "database_id": ["pubchem:6912387"],
            }
        ]
    # Use the iron-free monomer for BGC0001592 and add bagremycins
    # (see https://www.biorxiv.org/content/10.1101/631242v1.full-text)
    elif bgc_id == "BGC0001592":
        entry["compounds"] = [
            {
                "compound": "(4-ethenylphenyl) 4-hydroxy-3-nitrosobenzoate",
                "database_id": ["pubchem:193618"],
            },
            *(
                { "compound": f"bagremycin {x}" }
                for x in "ABCDEF"
            )
        ]
    # Use the iron-free chelator for BGC0001989
    # (pulcherriminic acid instead of of pucherrimin)
    elif bgc_id == "BGC0001989":
        entry["compounds"] = [
            {
                "compound": "pulcherriminic acid",
                "database_id": ["pubchem:3083664"],
            }
        ]
    # Use the iron-free chelator for BGC0002300
    elif bgc_id == "BGC0002300":
        entry["compounds"] = [
            {
                "compound": "deferrialbomycin delta2",
                "database_id": ["pubchem:86290046"],
            }
        ]
    # Use the copper-free chelator for BGC0002645
    elif bgc_id == "BGC0002645":
        entry["compounds"] = [
            {
                "compound": "fluopsin",
                "database_id": ["pubchem:3084535"],
            }
        ]
    # Use proper compounds for the partial fungal BGCs synthesizing dalmano
    # (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6335865/)
    elif bgc_id == "BGC0001906":
        entry["compounds"] = [
            {
                "compound": "naphthalene-1,3,6,8-tetrol",
                "database_id": ["pubchem:440202"],
            },
            {
                "compound": "2-acetylnaphthalene-1,3,6,8-tetrol",
                "database_id": ["pubchem:86018477"],
            },
        ]
    elif bgc_id == "BGC0001907":
        entry["compounds"] = [
            {
                "compound": "5-hydroxy-2-methyl-4-chromanone",
                "database_id": ["pubchem:13316455"],
            },
            {
                "compound": "1-(2,6-dihydroxyphenyl)but-2-en-1-one",
                "chem_struct": r"C1=C(O)C(C(=O)/C=C/C)=C(O)C=C1",
            },
        ]
    # pinensin A and B produced by cluster 
    # (see https://pubmed.ncbi.nlm.nih.gov/26211520/)
    elif bgc_id == "BGC0001392":
        entry["compounds"] = [
            {
                "compound": "pinensin A",
                "database_id": ["npatlas:NPA020235"]
            },
            {
                "compound": "pinensin B",
                "database_id": ["npatlas:NPA020236"]
            },
        ]
    # BGC0001575 also produced PZN4 (tyrvalin) as shown by heterologous 
    # expression (see doi:10.1016/j.cell.2016.12.021, Fig. 2)
    elif bgc_id == "BGC0001575":
        entry["compounds"].insert(
            0, 
            {
                "compound": "PZN4",
                "chem_struct": "C1=C(CC2=CC=C(O)C=C2)NC(=O)C(C(C)C)=N1",
            }
        )
    # BGC0001774 produces septacidin, not spicamycin
    # (see PMID:29483275)
    elif bgc_id == "BGC0001774":
        entry["compounds"] = [
            {
                "compound": "septacidin",
                "database_id": ["pubchem:96278"],
            }
        ]
    # BGC0000898 is only the subcluster producing desosamine, not the complete
    # pikromycin BGC (see https://github.com/mibig-secmet/mibig-json/issues/388),
    # and the desosamine formula in MIBIG is wrong:
    elif bgc_id == "BGC0000898":
        entry["compounds"] = [
            {
                "compound": "desosamine",
                "database_id": ["pubchem:168997"],
            }
        ]
    # BGC0002522 only produces pseudodesmin not viscosinamide
    # (see https://github.com/mibig-secmet/mibig-json/issues/377)
    elif bgc_id == "BGC0002522":
        entry["compounds"] = [
            {
                "compound": "pseudodesmin A",
                "database_id": ["npatlas:NPA013302"],
            }
        ]
    # BGC0000978 cylindrospermopsin has wrong cross-references to
    # PubChem (see https://github.com/mibig-secmet/mibig-json/issues/370)
    elif bgc_id == "BGC0000978":
        assert entry["compounds"][0]["compound"] == "cylindrospermopsin"
        entry["compounds"][0]["database_id"] = ["pubchem:42628600"]
    # BGC0000650 has wrong carotenoid link but actually (likely) produces
    # flexixanthin (see https://pubmed.ncbi.nlm.nih.gov/16625353/); since
    # crtZ is missing from the cluster we just add the deoxy- variant
    elif bgc_id == "BGC0000650":
        entry["compounds"] = [
            {
                 "compound": "deoxyflexixanthin",
                 "database_id": ["pubchem:6443734"],
            }
        ]
    # BGC0002426 is the "supercluster" encoding both borregomycins and
    # anthrabenzoxocinones, not the svetamycins as recorded in MIBiG
    # (see 10.1039/D1OB00600B, Fig.1, and Table S22, and also
    # https://github.com/mibig-secmet/mibig-json/issues/365)
    elif bgc_id == "BGC0002426":
        entry["compounds"] = [
            {"compound": "borregomycin B", "database_id": []},
            {"compound": "borregomycin C", "database_id": []},
            {"compound": "borregomycin E", "database_id": ["npatlas:NPA033080"]},
            {"compound": "borregomycin F", "database_id": ["npatlas:NPA033081"]},
            {"compound": "(-)-anthrabenzoxocinone A", "database_id": []},
            {"compound": "(-)-anthrabenzoxocinone B", "database_id": []},
            {"compound": "(-)-anthrabenzoxocinone C", "database_id": []},
            {"compound": "(-)-anthrabenzoxocinone D", "database_id": []},
            {"compound": "(-)-anthrabenzoxocinone E", "database_id": []},
            {"compound": "(+)-anthrabenzoxocinone G", "database_id": ["npatlas:NPA033082"]},
            {"compound": "(-)-anthrabenzoxocinone K", "database_id": ["npatlas:NPA033178"]},
            {"compound": "(-)-anthrabenzoxocinone X", "database_id": []},
        ]
    # BGC0000447 is only tolaasin F, not tolaasin I
    # (see https://github.com/mibig-secmet/mibig-json/issues/337)
    elif bgc_id == "BGC0000447":
        tolaasin_i = next(e for e in entry["compounds"] if e["compound"] == "tolaasin I")
        entry["compounds"].remove(tolaasin_i)
    # BGC0002681 compound is rhizoferrin, not rhizobactin
    # (see https://github.com/mibig-secmet/mibig-json/issues/290)
    elif bgc_id == "BGC0002681":
        assert len(entry["compounds"]) == 1
        entry["compounds"][0]["compound"] = "rhizoferrin"
    # BGC0001556 produces combamides A-E
    elif bgc_id == "BGC0001556":
        entry["compounds"] = [
            {"compound": f"combamide {x}"}
            for x in "ABCDE"
        ]
    # BGC0001452 produces sipanmycins A-B
    elif bgc_id == "BGC0001452":
        entry["compounds"] = [
            {"compound": f"sipanmycin {x}"}
            for x in "AB"
        ]
    # BGC0001824 produces xefoampeptides A-G
    elif bgc_id == "BGC0001824":
        entry["compounds"] = [
            {"compound": f"xefoampeptide {x}"}
            for x in "ABCDEFG"
        ]
    # BGC0001826 produces xeneprotides A-C
    elif bgc_id == "BGC0001826":
        entry["compounds"] = [
            {"compound": f"xeneprotide {x}"}
            for x in "ABC"
        ]
    # BGC0001650 produces Le-pyrrolopyrazines A-C
    elif bgc_id == "BGC0001650":
        entry["compounds"] = [
            {"compound": f"Le-pyrrolopyrazine {x}"}
            for x in "ABC"
        ]
    # BGC0001658 produces macrotermycins A-D
    elif bgc_id == "BGC0001658":
        entry["compounds"] = [
            {"compound": f"macrotermycin {x}"}
            for x in "ABCD"
        ]
    # BGC0001823 produces weishanmycins A1-3
    elif bgc_id == "BGC0001823":
        entry["compounds"] = [
            {"compound": f"weishanmycin A{x}"}
            for x in range(1, 4)
        ]
    # BGC0001627 produces isonitrile antibiotics described in the paper
    # (see https://pubmed.ncbi.nlm.nih.gov/28634299/, Fig. 2)
    elif bgc_id == "BGC0001627":
        entry["compounds"] = [
            {
                "compound": "isonitrile lipopeptide 1",
                "chem_struct": "C[C@@H]([N+]#[C-])CC(=O)NCCCC[C@@H](COC(C)=O)NC(=O)C[C@H]([N+]#[C-])C",
                "database_id": ["pubchem:137333763"],
            },
            {
                "compound": "isonitrile lipopeptide 2",
                "chem_struct": "C[C@@H]([N+]#[C-])CC(=O)NCCCC[C@@H](CO)NC(=O)C[C@H]([N+]#[C-])C",
                "database_id": ["pubchem:137333764"],
            }
        ]
    # BGC0001564 produces cusperins A and B
    # (see https://pubmed.ncbi.nlm.nih.gov/29570981/)
    elif bgc_id == "BGC0001564":
        entry["compounds"] = [
            {
                "compound": "cusperin A",
                "database_id": ["npatlas:NPA027290"],
            },
            {
                "compound": "cusperin B",
                "database_id": ["npatlas:NPA027291"],
            },
        ]
    # BGC0000261 produces R1128A-D compounds
    # (see https://pubmed.ncbi.nlm.nih.gov/10931852/)
    elif bgc_id == "BGC0000261":
        entry["compounds"] = [
            {"compound": "R1128A", "database_id": ["npatlas:NPA020125"]},
            {"compound": "R1128B", "database_id": ["npatlas:NPA020126"]},
            {"compound": "R1128C", "database_id": ["npatlas:NPA020127"]},
            {"compound": "R1128D", "database_id": ["npatlas:NPA020128"]},
        ]
    # BGC0001752 produces qinichelins
    # (see https://pmc.ncbi.nlm.nih.gov/articles/PMC5696649/)
    elif bgc_id == "BGC0001752":
        entry["compounds"] = [
            {"compound": "qinichelin", "database_id": ["npatlas:NPA028371"]},
            {"compound": "dehydroxy-qinichelin", "database_id": ["npatlas:NPA028374"]},
            {"compound": "acetyl-qinichelin 1", "database_id": ["npatlas:NPA028373"]},
            {"compound": "acetyl-qinichelin 2", "database_id": ["npatlas:NPA028372"]},
        ]
    # BGC0001762 produces rubrolone A-B
    # (see https://www.nature.com/articles/ncomms13083)
    elif bgc_id == "BGC0001762":
        entry["compounds"] = [
            {
                "compound": "rubrolone A",
                "chem_struct": "CCCC1=C2C(=CC(=N1)C)C3=C(C4=C(C(=O)C=C3C2=O)[C@@]5([C@@H]([C@@H]([C@H](O[C@@H]5O4)C)O)O)O)O",
                "database_id": ["npatlas:NPA021088"],
            },
            {
                "compound": "rubrolone B",
                "chem_struct": "CCCC1=C2C(=C3C(=CC(=O)C4=C(C3=O)O[C@H]5[C@@]4([C@H]([C@H]([C@H](O5)C)O)O)O)C2=O)C=C(N1C6=CC=CC=C6C(=O)O)C",
                "database_id": ["npatlas:NPA023798"],
            }
        ]
    # BGC0002666 produces legonmycins A-B and legonindolizines A-B
    # (see doi:10.1002/anie.201502902)
    elif bgc_id == "BGC0002666":
        entry["compounds"] = [
            {
                "compound": "legonmycin A",
                "chem_struct": "CC1=C(N2CCCC2(C1=O)O)NC(=O)CC(C)C",
                "database_id": ["npatlas:NPA007477"],
            },
            {
                "compound": "legonmycin B",
                "chem_struct": "CC1=C(NC(=O)CCC(C)C)N2CCCC2(O)C1=O",
                "database_id": ["npatlas:NPA032572"],
            },
            {
                "compound": "legonindolizine A",
                "chem_struct": "CC1=C(NC(CC(C)C)=O)C(N2CCCC2=C1O)=O",
                "database_id": [],
            },
            {
                "compound": "legonindolizine B",
                "chem_struct": "CC1=C(NC(CCC(C)C)=O)C(N2CCCC2=C1O)=O",
                "database_id": [],
            },
        ]
    # BGC0000431 produces stenothricin D
    # (see PMID:24149839)
    elif bgc_id == "BGC0000431":
        entry["compounds"] = [
            {
                "compound": "stenothricin D",
                "database_id": ["npatlas:NPA007735"],
            }
        ]
    # BGC0000291 produces all A-54145 lipopeptide variants
    elif bgc_id == "BGC0000291":
        entry["compounds"] = [
            {"compound": f"A-54145 {x}"}
            for x in ["A", "A1", "B", "B1", "C", "D", "E", "F"]
        ]
    # BGC0001574 produces compound named SF2768
    elif bgc_id == "BGC0001574":
        entry["compound"] = [
            {
                "chem_struct": "[C-]#[N+]C(C)CC(=O)NCC1CCC(NC(=O)CC(C)[N+]#[C-])C(O)O1",
                "compound": "SF2768",
                "database_id": ["npatlas:NPA028494"],
            }
        ]

    for compound in entry["compounds"]:
        # mask formula of all capsular polysaccharide BGCs
        if compound["compound"] == "capsular polysaccharide":
            compound.pop("chem_struct", None)
            break
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
            compound["chem_struct"] = r"C[N+]1(C)CC2=C(C1)C1=C(NC3=C1C=CC=C3)C1=C2C2=C(N1)C=CC=C2"
        # add formula of bacillothiazole A
        elif compound["compound"] == "bacillothiazol A":
            compound["chem_struct"] = "S1C(CCCCC(C)CC)=NC(C2=NC(C3SC=C(C(N[C@@](CO)C(=O)O)=O)N=3)=CS2)=C1"
        # add aspcandine
        elif compound["compound"] == "aspcandine":
            compound["chem_struct"] = "C1[C@@]([H])2NC(=O)C=C2NC2=C(O)C=CC=C2C1=O"
        # add 3-thiaglutamate formula
        elif compound["compound"] == "3-thiaglutamate":
            compound["chem_struct"] = "NC(C(=O)O)SCC(=O)O"
        # fix clipibicyclene annotation
        elif compound["compound"] == "clipibycyclene":
            compound["compound"] = "clipibicyclene"
            compound["chem_struct"] = r"C1=C(/NC(=O)/C=C/C(/C)=C/C=C/C(O)CC)\OC(=O)N2CC(O)/C/2=C/1.O.N"
        elif compound["compound"] == "thermochelin":
            compound["chem_struct"] = r"CC(N(CCCC(NC(=O)C)C(=O)NC(CCC(N)=O)C(N(O)CCCC1NC(=O)C(CCCN(O)C(C)=O)NC1=O)=O)O)=O"
        # fix coronofacic acid
        elif compound["compound"] == "oronofacic acid":
            compound["compound"] = "coronofacic acid"
        # assign propert variants
        elif compound["compound"] == "guangnanmycin":
            compound["compound"] = "guangnanmycin A"
        elif compound["compound"] == "entolysin":
            compound["compound"] = "entolysin A"
        elif compound["compound"] == "kolossin":
            compound["compound"] = "kolossin A"
        elif compound["compound"] == "nenestatin":
            compound["compound"] = "nenestatin A"
        elif compound["compound"] == "ochrobactin":
            compound["compound"] = "ochrobactin A"
        elif compound["compound"] == "ulleungmycin":
            compound["compound"] = "ulleungmycin A"
        elif compound["compound"] == "rimosamide":
            compound["compound"] = "rimosamide A"
        elif compound["compound"] == "tridecaptin":
            compound["compound"] = "tridecaptin A1"
        elif compound["compound"] == "putisolvin":
            compound["compound"] = "putisolvin I"
        # exochelin has name exochelin MS in NPAtlas
        elif compound["compound"] == "exochelin":
            compound["compound"] = "exochelin MS"


# --- Load NPAtlas -----------------------------------------------------------
with rich.progress.open(args.atlas, "rb", description=f"[bold blue]{'Loading':>12}[/] NPAtlas") as handle:
    data = json.load(gzip.open(handle))

np_atlas = {
    entry["npaid"]: entry
    for entry in rich.progress.track(data, description="Indexing NPAtlas...")
}
del data

# --- Replace compounds from MIBiG that have a NPAtlas annotation ------------

#mibig_entries = collections.defaultdict(list)
#for npaid, entry in rich.progress.track(np_atlas.items(), description="Patching MIBiG..."):
#    mibig_xrefs = [xref["external_db_code"] for xref in entry["external_ids"] if xref["external_db_name"] == "mibig"]
#    for xref in mibig_xrefs:
#        mibig_entries[xref].append(entry)

#for bgc_id, bgc in mibig.items():
#    if bgc_id in mibig_entries and len(bgc["compounds"]) == 1 and "chem_struct" not in bgc["compounds"][0]:
#        names = ", ".join(repr(compound["compound"]) for compound in bgc['compounds'])
#        npnames = ", ".join(repr(entry['original_name']) for entry in mibig_entries[bgc_id])
#        rich.print(f"[bold blue]{'Replacing':>12}[/] compounds of [purple]{bgc_id}[/] ({names} with {npnames})")
#        bgc['compounds'] = [
#            {
#                "compound": entry["original_name"],
#                "chem_struct": entry["smiles"],
#                "database_id": [f"npatlas:{entry['npaid']}"],
#            }
#            for entry in mibig_entries[bgc_id]
#        ]

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
                compound["database_id"] = [f"pubchem:{cids[0]}"]
                compound["chem_struct"] = c.isomeric_smiles
                rich.print(f"[bold green]{'Mapped':>12}[/] {compound['compound']!r} product of [purple]{entry['mibig_accession']}[/] to PubChem compound {c.cid}")
                time.sleep(1)
            else:
                rich.print(f"[bold red]{'Failed':>12}[/] to map {compound['compound']!r} product of [purple]{entry['mibig_accession']}[/] to PubChem")

# --- Retrieve SMILES for compound with a cross-reference --------------------

for bgc_id, entry in rich.progress.track(mibig.items(), description=f"[bold blue]{'Downloading':>12}[/]"):
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
