import itertools
import json
import time
import uuid
import urllib.request
import urllib.parse
from typing import Dict, Set, List

import numpy
import pandas

from .predictor import ChemicalHierarchyPredictor
from .treematrix import TreeMatrix

_BASE_URL = "http://classyfire.wishartlab.com"


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


def binarize_classification(classes: pandas.DataFrame, hierarchy: TreeMatrix, leaves: Set[str]) -> numpy.ndarray:
    out = numpy.zeros(len(classes), dtype=bool)

    indices = []
    for leaf in leaves:
        try:
            index = classes.index.get_loc(leaf)
            out[index] = True
        except KeyError:
            pass

    for i in reversed(hierarchy):
        if out[i]:
            out[hierarchy.parents(i)] = True

    return out