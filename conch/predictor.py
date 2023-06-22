import contextlib
import json
import math
import pickle
import typing
import importlib.resources
from typing import Any, Literal, List, Tuple, Dict, TextIO, Type, Union, Optional, Callable

import anndata
import numpy
import pandas
import scipy.sparse
import sklearn.multiclass
import sklearn.linear_model
from scipy.special import expit

from . import _json
from .treematrix import TreeMatrix

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

_T = typing.TypeVar("_T", bound="ChemicalHierarchyPredictor")

class ChemicalHierarchyPredictor:
    """A model for predicting chemical hierarchy from BGC compositions.
    """

    def __init__(self, hierarchy, n_jobs=None, max_iter=100):
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.classes_ = None
        self.features_ = None
        self.coef_ = None
        self.intercept_ = None
        self.hierarchy = hierarchy

    def __getstate__(self):
        return {
            "classes_": self.classes_,
            "features_": self.features_,
            "intercept_": list(self.intercept_),
            "hierarchy": self.hierarchy.__getstate__(),
            "coef_": scipy.sparse.csr_matrix(self.coef_),
        }

    def __setstate__(self, state):
        self.classes_ = state["classes_"]
        self.features_ = state["features_"]
        self.intercept_ = numpy.asarray(state["intercept_"])
        self.coef_ = state["coef_"].toarray()
        self.hierarchy.__setstate__(state["hierarchy"])

    def fit(self: _T, X, Y) -> _T:
        if isinstance(X, anndata.AnnData):
            _X = X.X
            self.features_ = X.var
        else:
            _X = X
            self.features_ = pandas.DataFrame(index=list(map(str, range(1, _X.shape[1] + 1))))
        if isinstance(Y, anndata.AnnData):
            _Y = Y.X
            self.classes_ = Y.var
        else:
            _Y = Y
            self.classes_ = pandas.DataFrame(index=list(map(str, range(1, _Y.shape[1] + 1))))

        # train model using scikit-learn
        model = sklearn.multiclass.OneVsRestClassifier(
            sklearn.linear_model.LogisticRegression("l1", solver="liblinear", max_iter=self.max_iter),
            n_jobs=self.n_jobs,
        ).fit(_X, _Y)

        # copy coefficients & intercept to a single NumPy array
        self.coef_ = numpy.zeros((_X.shape[1], _Y.shape[1]), order="C")
        self.intercept_ = numpy.zeros(_Y.shape[1], order="C")
        for i, estimator in enumerate(model.estimators_):
            if isinstance(estimator, sklearn.linear_model.LogisticRegression):
                self.coef_[:, i] = estimator.coef_
                self.intercept_[i] = estimator.intercept_
            else:
                self.intercept_[i] = -1000 if estimator.y_[0] == 0 else 1000

        # remove features with all-zero weights
        nonzero_weights = numpy.abs(self.coef_).sum(axis=1) > 0
        self.coef_ = self.coef_[nonzero_weights]
        self.features_ = self.features_[nonzero_weights]

        return self

    def propagate(self, Y) -> numpy.ndarray:
        _Y = numpy.array(Y, dtype=bool)
        for i in reversed(self.hierarchy):
            for j in self.hierarchy.parents(i):
                _Y[:, j] |= _Y[:, i]
        return _Y

    def predict_probas(self, X) -> numpy.ndarray:
        if isinstance(X, anndata.AnnData):
            _X = X.X.toarray()
        else:
            _X = numpy.asarray(X)
        return expit(_X @ self.coef_ + self.intercept_)

    def predict(self, X, propagate=True) -> numpy.ndarray:
        probas = self.predict_probas(X)
        classes = probas > 0.5
        if propagate:
            return self.propagate(classes)
        return classes

    def save(self, file: TextIO) -> None:
        state = self.__getstate__()
        json.dump(state, file, cls=_json.JSONEncoder, sort_keys=True, indent=1)
        # pickle.dump(state, file)

    @classmethod
    def trained(cls: Type[_T]) -> _T:
        with files(__package__).joinpath("predictor.json").open() as f:
            return cls.load(f)

    @classmethod
    def load(cls: Type[_T], file: TextIO) -> _T:
        state = json.load(file, cls=_json.JSONDecoder)
        predictor = cls(TreeMatrix())
        predictor.__setstate__(state)
        return predictor

