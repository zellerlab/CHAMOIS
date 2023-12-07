import dataclasses
import itertools
import json
import gzip
import time
import pathlib
import shutil
import uuid
import urllib.request
import urllib.parse
from typing import Dict, Set, List

import platformdirs
import numpy
import pandas

from .ontology import IncidenceMatrix
from .predictor import ChemicalOntologyPredictor

_BASE_URL = "http://classyfire.wishartlab.com"


@dataclasses.dataclass(frozen=True)
class Term:
    id: str
    name: str
    description: str

    def __eq__(self, other: object):
        if not isinstance(other, Term):
            return NotImplemented
        return other.id == self.id

    def __hash__(self) -> int:
        return hash(self.id)

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "Term":
        return cls(id=d["chemont_id"], name=d["name"], description=d["description"])

    @property
    def url(self):
        prefix, local = self.id.split(":")
        return f"http://classyfire.wishartlab.com/tax_nodes/{local}"


@dataclasses.dataclass(frozen=True)
class Classification:
    alternative_parents: List[Term]
    ancestors: List[str]
    class_: Term
    classification_version: str
    description: str
    direct_parent: Term
    inchikey: str
    intermediate_nodes: List[Term]
    kingdom: Term
    molecular_framework: str
    predicted_chebi_terms: List[str]
    predicted_lipidmaps_terms: List[str]
    smiles: str
    subclass: Term
    substituents: List[str]
    superclass: Term

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "Classification":
        return cls(
            alternative_parents=[Term.from_dict(x) for x in d.get('alternative_parents', [])],
            ancestors=d['ancestors'],
            class_=None if d['class'] is None else Term.from_dict(d['class']),
            classification_version=d['classification_version'],
            description=d['description'],
            direct_parent=Term.from_dict(d['direct_parent']),
            inchikey=d['inchikey'],
            intermediate_nodes=[Term.from_dict(x) for x in d.get('intermediate_nodes', [])],
            kingdom=None if d['kingdom'] is None else Term.from_dict(d['kingdom']),
            molecular_framework=d['molecular_framework'],
            predicted_chebi_terms=d['predicted_chebi_terms'],
            predicted_lipidmaps_terms=d['predicted_lipidmaps_terms'],
            smiles=d['smiles'],
            subclass=None if d['subclass'] is None else Term.from_dict(d['subclass']),
            substituents=d['substituents'],
            superclass=Term.from_dict(d['superclass']),
        )

    @property
    def terms(self) -> Set[Term]:
        return set(filter(None, (
            *self.alternative_parents,
            self.kingdom,
            self.superclass,
            self.class_,
            self.subclass,
            self.direct_parent,
            *self.intermediate_nodes,
        )))


def get_classification(inchikey: str) -> Classification:
    cache = pathlib.Path(platformdirs.user_cache_dir('CONCH', 'ZellerLab'))
    entry = cache.joinpath(inchikey).with_suffix(".json.gz") 
    if not entry.exists():
        cache.mkdir(parents=True, exist_ok=True)
        url = f"https://cfb.fiehnlab.ucdavis.edu/entities/{inchikey}.json"
        with urllib.request.urlopen(url) as res:
            response = json.load(res)
        if "error" in response:
            raise RuntimeError(f"Failed to get classification: {response['error']}")
        with gzip.open(entry, "wt") as dst:
            json.dump(response, dst)
    with gzip.open(entry, "rt") as f:
        return Classification.from_dict(json.load(f))


def query_classyfire(structures: List[str]) -> Dict[str, object]:
    form = {
        "label": f"conch-{uuid.uuid4()}",
        "query_input": "\n".join(structures),
        "query_type": "STRUCTURE",
    }
    request = urllib.request.Request(
        f"{_BASE_URL}/queries.json",
        data=json.dumps(form).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request) as res:
        return json.load(res)


def get_results(query_id: int, page: int = 1) -> Dict[str, object]:
    request = urllib.request.Request(
        f"{_BASE_URL}/queries/{query_id}.json?page={page}",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as res:
        return json.load(res)


def extract_classification(data: Dict[str, object]) -> Set[str]:
    return {
        direct_parent["chemont_id"]
        for direct_parent in itertools.chain(
            [data[x] for x in ["kingdom", "superclass", "class", "subclass", "direct_parent"]],
            data["intermediate_nodes"],
            data["alternative_parents"],
        )
        if direct_parent is not None
    }


def binarize_classification(
    classes: pandas.DataFrame, 
    incidence_matrix: IncidenceMatrix, 
    leaves: Set[str]
) -> numpy.ndarray:
    out = numpy.zeros(len(classes), dtype=bool)

    indices = []
    for leaf in leaves:
        try:
            index = classes.index.get_loc(leaf)
            out[index] = True
        except KeyError:
            pass

    for i in reversed(incidence_matrix):
        if out[i]:
            out[incidence_matrix.parents(i)] = True

    return out