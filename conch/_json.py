import json
from typing import Any

import numpy
import pandas
import scipy.sparse


class JSONEncoder(json.JSONEncoder):

    def default(self, o: Any) -> object:
        if isinstance(o, numpy.integer):
            return int(o)
        elif isinstance(o, numpy.floating):
            return float(o)
        elif isinstance(o, numpy.ndarray):
            return list(o)
        elif isinstance(o, numpy.bool_):
            return bool(o)
        elif isinstance(o, pandas.DataFrame):
            return dict(__type__="DataFrame", **o.to_dict(orient="split"))
        elif isinstance(o, scipy.sparse.csr_matrix):
            return dict(
                __type__="csr_matrix",
                shape=o.shape,
                data=o.data,
                indices=o.indices,
                indptr=o.indptr,
            )
        return super().default(o)


class JSONDecoder(json.JSONDecoder):

    def __init__(self):
        super().__init__(object_hook=self._hook)

    @staticmethod
    def _hook(obj):
        ty = obj.pop("__type__", None)
        if ty == "DataFrame":
            return pandas.DataFrame(**obj)
        elif ty == "csr_matrix":
            return scipy.sparse.csr_matrix(
                (obj["data"], obj["indices"], obj["indptr"]), 
                shape=obj["shape"]
            )
        return obj
