import argparse
import itertools
import json
import gzip
import os

import anndata
import joblib
import numpy
import pandas
import pronto
import rich.progress
import scipy.sparse
import rdkit.Chem
from rdkit.Chem import MACCSkeys

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
parser.add_argument("--atlas", required=True)
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()


# --- Load NPAtlas ------------------------------------------------------------

with rich.progress.open(args.atlas, "rb", description=f"[bold blue]{'Unzipping':>12}[/]") as handle:
    data = json.load(gzip.open(handle))

rich.print(f"[bold green]{'Loaded':>12}[/] {len(data)} compounds from NPAtlas")
np_atlas = {entry["npaid"]: entry for entry in data}
np_atlas_indices = {entry["npaid"]:i for i, entry in enumerate(data)}


# --- Binarize NPAtlas classes ------------------------------------------------

maccs = numpy.zeros((len(np_atlas), 167), dtype=numpy.bool_)

for i, compound in enumerate(rich.progress.track(np_atlas.values(), description=f"[bold blue]{'Binarizing':>12}[/]")):
    mol = rdkit.Chem.MolFromSmiles(compound["smiles"])
    maccs[i] = numpy.array(MACCSkeys.GenMACCSKeys(mol), dtype=numpy.bool_)

# --- Store metadata ----------------------------------------------------------

np_classes = anndata.AnnData(
    X=scipy.sparse.csr_matrix(maccs),
    var=pandas.DataFrame(
        index=range(len(NAMES)),
        data=dict(name=NAMES),
    ),
    obs=pandas.DataFrame(
        index=list(np_atlas_indices),
        data=dict(
            compound=[entry["original_name"] for entry in data],
            inchikey=[entry["inchikey"] for entry in data],
            smiles=[entry["smiles"] for entry in data],
        )
    ),
    dtype=numpy.bool_
)

# save annotated data
os.makedirs(os.path.dirname(args.output), exist_ok=True)
np_classes.write(args.output)
