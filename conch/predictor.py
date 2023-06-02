import contextlib
import json
import math
import pickle
import typing
import importlib.resources
from typing import Any, Literal, List, Tuple, Dict, BinaryIO, Type, Union, Optional, Callable

import numpy
import pandas
import scipy.sparse
import sklearn.multiclass
import sklearn.linear_model
from scipy.special import expit

try:
    import anndata
except ImportError:
    anndata = None

_T = typing.TypeVar("_T", bound="ChemicalHierarchyPredictor")

class ChemicalHierarchyPredictor:
    """A model for predicting chemical hierarchy from BGC compositions.
    """

    def __init__(self, n_jobs=None, hierarchy=None, max_iter=100):
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
            "intercept_": self.intercept_,
            "coef_": scipy.sparse.csr_matrix(self.coef_),
            "hierarchy": self.hierarchy,
        }

    def __setstate__(self, state):
        self.classes_ = state["classes_"]
        self.features_ = state["features_"]
        self.intercept_ = numpy.asarray(state["intercept_"])
        self.coef_ = state["coef_"].toarray()
        self.hierarchy = state["hierarchy"]

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

        model = sklearn.multiclass.OneVsRestClassifier(
            sklearn.linear_model.LogisticRegression("l1", solver="liblinear", max_iter=self.max_iter),
            n_jobs=self.n_jobs,
        ).fit(_X, _Y)

        coef_ = numpy.zeros((_X.shape[1], _Y.shape[1]), order="F")
        intercept_ = numpy.zeros(_Y.shape[1], order="F")

        for i, estimator in enumerate(model.estimators_):
            if isinstance(estimator, sklearn.linear_model.LogisticRegression):
                coef_[:, i] = estimator.coef_
                intercept_[i] = estimator.intercept_
            else:
                intercept_[i] = -1000 if estimator.y_[0] == 0 else 1000

        self.coef_ = scipy.sparse.csr_matrix(coef_)
        self.intercept_ = intercept_
        return self

    def predict_proba(self, X) -> numpy.ndarray:
        if isinstance(X, anndata.AnnData):
            _X = X.X.toarray()
        else:
            _X = numpy.asarray(X)
        return expit(_X @ self.coef_ + self.intercept_)

    def save(self, file: BinaryIO) -> None:
        state = self.__getstate__()
        pickle.dump(state, file)
        # numpy.savez_compressed(file, **state)

    @classmethod
    def trained(cls: Type[_T]) -> _T:
        with importlib.resources.open_binary("conch", "predictor.pkl") as f:
            return cls.load(f)

    @classmethod
    def load(cls: Type[_T], file: BinaryIO) -> _T:
        predictor = cls()
        state = pickle.load(file)
        predictor.__setstate__(state)
        return predictor

    # def save(self, file: BinaryIO) -> None:
    #     state = self.__getstate__()
    #     torch.save(state, file)