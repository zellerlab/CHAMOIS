"""HTTP client for the REST API of ClassyFire.
"""

import dataclasses
import itertools
import json
import gzip
import time
import os
import pathlib
import shutil
import uuid
import urllib.request
import urllib.parse
import typing
from urllib.request import Request
from typing import Dict, Set, List, Iterable, Optional, Union

import platformdirs
import numpy

from .ontology import AdjacencyMatrix
from .predictor import ChemicalOntologyPredictor

if typing.TYPE_CHECKING:
    from pandas import DataFrame

_BASE_URL = "http://classyfire.wishartlab.com"


@dataclasses.dataclass(frozen=True)
class Term:
    """A ChemOnt term.
    """
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
        """Get a `Term` from a raw dictionary returned by ClassyFire.
        """
        return cls(id=d["chemont_id"], name=d["name"], description=d["description"])

    @property
    def url(self):
        """`str`: The URL for this term.
        """
        prefix, local = self.id.split(":")
        return f"http://classyfire.wishartlab.com/tax_nodes/{local}"


@dataclasses.dataclass(frozen=True)
class Classification:
    """A complete ClassyFire classification for a given molecule.
    """
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
        """Get a `Classification` from a raw dictionary returned by ClassyFire.
        """
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
        """`set` of `Term`: The set of leaf terms in the classification.
        """
        return set(filter(None, (
            *self.alternative_parents,
            self.kingdom,
            self.superclass,
            self.class_,
            self.subclass,
            self.direct_parent,
            *self.intermediate_nodes,
        )))


class Cache(typing.MutableMapping[str, Dict[str, object]]):
    """A local, filesystem-backed cache for ClassyFire results.

    Attributes:
        folder (`~pathlib.Path`): The path to the folder where the cache
            data is located.

    """

    def __init__(self, folder: "Optional[os.PathLike[str]]" = None):
        if folder is None:
            self.folder = pathlib.Path(platformdirs.user_cache_dir('CHAMOIS', 'ZellerLab')).joinpath("classyfire")
        else:
            self.folder = pathlib.Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return f"{type(self).__name__}({self.folder!r})"

    def __getitem__(self, item: str) -> Dict[str, object]:
        entry = self.folder.joinpath(item).with_suffix(".json.gz")
        try:
            with gzip.open(entry, "rt") as src:
                return json.load(src)
        except FileNotFoundError as err:
            raise KeyError(item) from None

    def __setitem__(self, item: str, value: Dict[str, object]) -> None:
        entry = self.folder.joinpath(item).with_suffix(".json.gz")
        with gzip.open(entry, "wt") as dst:
            json.dump(value, dst)

    def __delitem__(self, item: str) -> None:
        entry = self.folder.joinpath(item).with_suffix(".json.gz")
        if not entry.exists():
            raise KeyError(item)
        os.remove(entry)

    def __iter__(self) -> Iterable[str]:
        return (path.stem[:-5] for path in self.folder.glob("*.json.gz"))

    def __len__(self) -> int:
        return sum(1 for _ in self.folder.glob("*.json.gz"))

    def __contains__(self, item: str):
        entry = self.folder.joinpath(item).with_suffix(".json.gz")
        return entry.exists()


class Query:
    """A query sent to ClassyFire.

    Attributes:
        client (`~chamois.classyfire.Client`): The ClassyFire client
            used to submit the query.
        id (`str`): The identifier given to the query by ClassyFire.

    """
    
    def __init__(self, client: "Client", id: str):
        self.id = id
        self.client = client

    @property
    def status(self) -> str:
        """`str`: The query status polled from the ClassyFire server.
        """
        request = Request(
            f"{self.client.base_url}/queries/{self.id}.json",
            headers={"Content-Type": "application/json"},
        )
        data = self.client._get(request)
        return data['classification_status']


class Client:

    def __init__(
        self,
        base_url: str = "http://classyfire.wishartlab.com",
        entities_url: str = "https://cfb.fiehnlab.ucdavis.edu/entities/",
        cache: Optional[Cache] = None,
        delay: float = 10.0,
    ):
        self.base_url = base_url
        self.entities_url = entities_url
        self.cache = Cache() if cache is None else cache
        self.delay = delay
        self._last_query = 0.0

    def _get(self, request: Union[str, Request]) -> Dict[str, object]:
        t = time.time()
        dt = t - self._last_query
        if dt < self.delay:
            time.sleep(self.delay - dt)
        self._last_query = t
        with urllib.request.urlopen(request) as res:
            response = json.load(res)
        return response

    def fetch(self, inchikey: str) -> Classification:
        """Fetch the pre-computed classification for a single compound.
        """
        if inchikey not in self.cache:
            url = f"{self.entities_url}{inchikey}.json"
            try:
                response = self._get(url)
                if 'error' in response:
                    raise RuntimeError(f"Failed to get classification: {response['error']}")
                elif 'report' in response:
                    raise RuntimeError(f"Failed to get classification: {' '.join(response['report'])}")
                elif not response:
                    raise RuntimeError(f"Failed to get classification")
            except urllib.error.HTTPError as err:
                raise RuntimeError(f"Failed to get classification: ClassyFire server down") from err
            self.cache[inchikey] = response
        return Classification.from_dict(self.cache[inchikey])

    def query(self, structures: Iterable[str]) -> Query:
        """Query ClassyFire with the given molecules.
        """
        form = {
            "label": f"chamois-{uuid.uuid4()}",
            "query_input": "\n".join(structures),
            "query_type": "STRUCTURE",
        }
        request = Request(
            f"{self.base_url}/queries.json",
            data=json.dumps(form).encode(),
            headers={"Content-Type": "application/json"}
        )
        query = self._get(request)
        if "id" not in query:
            raise RuntimeError("Failed to submit queries to ClassyFire")
        return Query(self, query["id"])

    def retrieve(self, query: Query, page: int = 1) -> Iterable[Classification]:
        """Retrieve the ClassyFire results for a query.
        """
        request = Request(
            f"{self.base_url}/queries/{query.id}.json?page={page}",
            headers={"Content-Type": "application/json"},
        )
        result = self._get(request)
        for entity in result.get('entities', []):
            inchikey = entity["inchikey"].split("=")[-1]
            self.cache[inchikey] = entity
        return result


def binarize_classification(
    classes: "DataFrame", 
    adjacency_matrix: AdjacencyMatrix, 
    leaves: Set[str]
) -> numpy.ndarray:
    """Convert a set of class names into a binary indicator vector.
    """
    out = numpy.zeros(len(classes), dtype=bool)

    indices = []
    for leaf in leaves:
        try:
            index = classes.index.get_loc(leaf)
            out[index] = True
        except KeyError:
            pass

    for i in reversed(adjacency_matrix):
        if out[i]:
            out[adjacency_matrix.parents(i)] = True

    return out
