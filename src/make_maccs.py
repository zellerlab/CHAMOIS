import argparse
import collections
import itertools
import json
import gzip
import os
import urllib.request
import time
from urllib.error import HTTPError
from typing import Dict, List

import anndata
import disjoint_set
import pandas
import numpy
import rich.progress
import scipy.sparse
import scipy.spatial.distance
import rdkit.Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder
from rdkit import RDLogger

NAMES = [
    'ISOTOPE',
    'atomic num >103',
    '104',
    'Group IVa,Va,VIa Rows 4-6',
    'actinide',
    'Group IIIB,IVB',
    'Lanthanide',
    'Group VB,VIB,VIIB',
    'QAAA@1',
    'Group VIII',
    'Group IIa',
    '4M Ring',
    'Group IB,IIB',
    'ON(C)C',
    'S-S',
    'OC(O)O',
    'QAA@1',
    'CTC',
    'Group IIIA',
    '7M Ring',
    'Si',
    'C=C(Q)Q',
    '3M Ring',
    'NC(O)O',
    'N-O',
    'NC(N)N',
    'C$=C($A)$A',
    'I',
    'QCH2Q',
    'P',
    'CQ(C)(C)A',
    'QX',
    'CSN',
    'NS',
    'CH2=A',
    'Group IA',
    'S Heterocycle',
    'NC(O)N',
    'NC(C)N',
    'OS(O)O',
    'S-O',
    'CTN',
    'F',
    'QHAQH',
    'OTHER',
    'C=CN',
    'BR',
    'SAN',
    'OQ(O)O',
    'CHARGE',
    'C=C(C)C',
    'CSO',
    'NN',
    'QHAAAQH',
    'QHAAQH',
    'OSO',
    'ON(O)C',
    'O Heterocycle',
    'QSQ',
    'Snot%A%A',
    'S=O',
    'AS(A)A',
    'A$!A$A',
    'N=O',
    'A$A!S',
    'C%N',
    'CC(C)(C)A',
    'QS',
    'QHQH',
    'QQH',
    'QNQ',
    'NO',
    'OAAO',
    'S=A',
    'CH3ACH3',
    'A!N$A',
    'C=C(A)A',
    'NAN',
    'C=N',
    'NAAN',
    'NAAAN',
    'SA(A)A',
    'ACH2QH',
    'QAAAA@1',
    'NH2',
    'CN(C)C',
    'CH2QCH2',
    'X!A$A',
    'S',
    'OAAAO',
    'QHAACH2A',
    'QHAAACH2A',
    'OC(N)C',
    'QCH3',
    'QN',
    'NAAO',
    '5 M ring',
    'NAAAO',
    'QAAAAA@1',
    'C=C',
    'ACH2N',
    '8M to 14M Ring',
    'QO',
    'CL',
    'QHACH2A',
    'A$A($A)$A',
    'QA(Q)Q',
    'XA(A)A',
    'CH3AAACH2A',
    'ACH2O',
    'NCO',
    'NACH2A',
    'AA(A)(A)A',
    'Onot%A%A',
    'CH3CH2A',
    'CH3ACH2A',
    'CH3AACH2A',
    'NAO',
    'ACH2CH2A > 1',
    'N=A',
    'Heterocyclic atom > 1',
    'N Heterocycle',
    'AN(A)A',
    'OCO',
    'QQ',
    'Aromatic Ring > 1',
    'A!O!A',
    'A$A!O > 1',
    'ACH2AAACH2A',
    'ACH2AACH2A',
    'QQ > 1',
    'QH > 1',
    'OACH2A',
    'A$A!N',
    'X (HALOGEN)',
    'Nnot%A%A',
    'O=A>1',
    'Heterocycle',
    'QCH2A>1',
    'OH',
    'O > 3',
    'CH3 > 2',
    'N > 1',
    'A$A!O',
    'Anot%A%Anot%A',
    '6M ring > 1',
    'O > 2',
    'ACH2CH2A',
    'AQ(A)A',
    'CH3 > 1',
    'A!A$A!A',
    'NH',
    'OC(C)C',
    'QCH2A',
    'C=O',
    'A!CH2!A',
    'NA(A)A',
    'C-O',
    'C-N',
    'O>1',
    'CH3',
    'N',
    'Aromatic',
    '6M Ring',
    'O',
    'Ring',
    'Fragments',
]

# disable logging
RDLogger.DisableLog('rdApp.warning')  

# get paths from command line
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
parser.add_argument("-o", "--output", required=True)
parser.add_argument("-D", "--distance", type=float, default=0.5)
args = parser.parse_args()


# --- Load MIBiG -------------------------------------------------------------

with rich.progress.open(args.input, "rb", description=f"[bold blue]{'Loading':>12}[/] BGCs") as handle:
    compounds = json.load(handle)

bgc_ids = sorted(compounds)
bgc_indices = {name:i for i, name in enumerate(bgc_ids)}

rich.print(f"[bold green]{'Loaded':>12}[/] {len(bgc_indices)} BGCs")


# --- Get ClassyFire annotations for all compounds ----------------------------

unknown_structure = numpy.zeros(len(bgc_ids), dtype=numpy.bool_)
maccs_keys = numpy.zeros((len(bgc_ids), 167), dtype=numpy.bool_)
names = [""] * len(bgc_ids)
smiles = [""] * len(bgc_ids)
inchikey = [""] * len(bgc_ids)

for bgc_id, bgc_compounds in rich.progress.track(compounds.items(), description=f"[bold blue]{'Classifying':>12}[/]"):
    bgc_index = bgc_indices[bgc_id]
    for compound in bgc_compounds:
        if "chem_struct" not in compound:
            # ignore compounds without structure (should have gotten one already)
            rich.print(f"[bold yellow]{'Skipping':>12}[/] {compound['compound']!r} compound of [bold cyan]{bgc_id}[/] with no structure")
        else:
            # compute MACCS key
            mol = rdkit.Chem.MolFromSmiles(compound["chem_struct"])
            maccs_keys[bgc_index] = numpy.array(MACCSkeys.GenMACCSKeys(mol), dtype=numpy.bool_)
            # record smiles and compound
            names[bgc_index] = compound["compound"]
            smiles[bgc_index] = compound["chem_struct"]
            inchikey[bgc_index] = rdkit.Chem.MolToInchiKey(mol)
            break
    else:
        unknown_structure[bgc_index] = True


# --- Build groups using MHFP6 distances -------------------------------------

rich.print(f"[bold blue]{'Building':>12}[/] MHFP6 fingerprints for {len(bgc_indices)} compounds")

encoder = MHFPEncoder(2048, 42)
group_set = disjoint_set.DisjointSet({ i:i for i in range(len(bgc_ids)) })
fps = numpy.array(encoder.EncodeSmilesBulk(smiles, kekulize=True))
indices = itertools.combinations(range(len(bgc_ids)), 2)
total = len(bgc_ids) * (len(bgc_ids) - 1) / 2

for (i, j) in rich.progress.track(indices, total=total, description=f"[bold blue]{'Joining':>12}[/]"):
    if not unknown_structure[i] and not unknown_structure[j]:
        d = scipy.spatial.distance.hamming(fps[i], fps[j])
        if d < args.distance:
            group_set.union(i, j)

n = sum(1 for _ in group_set.itersets())
rich.print(f"[bold green]{'Built':>12}[/] {n} groups of molecules with MHFP6 distance over {args.distance}")


# --- Create annotated data --------------------------------------------------

# generate annotated data
data = anndata.AnnData(
    dtype=numpy.bool_,
    X=scipy.sparse.csr_matrix(maccs_keys),
    obs=pandas.DataFrame(
        index=bgc_ids,
        data=dict(
            unknown_structure=unknown_structure,
            compound=names,
            smiles=smiles,
            inchikey=inchikey,
            groups=[group_set[i] for i in range(len(bgc_ids))]
        ),
    ),
    var=pandas.DataFrame(
        index=range(len(NAMES)),
        data=dict(name=NAMES),
    ),
)

# save annotated data
os.makedirs(os.path.dirname(args.output), exist_ok=True)
data.write(args.output)
 

