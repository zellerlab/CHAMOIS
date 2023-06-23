import itertools
import json
import time
from urllib.request import urlopen
from typing import Dict, Set

import numpy
import pandas

from .predictor import ChemicalHierarchyPredictor
from .treematrix import TreeMatrix


def query_classyfire(inchikey, wait: float = 10.0) -> Dict[str, object]:
    # otherwise use the ClassyFire website API
    with urlopen(f"http://classyfire.wishartlab.com/entities/{inchikey}.json") as res:
        data = json.load(res)
        if wait:
            time.sleep(10.0)
        if "class" not in data:
            raise RuntimeError("classification not found")
        return data


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