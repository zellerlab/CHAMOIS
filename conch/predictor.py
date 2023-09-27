import contextlib
import hashlib
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
from scipy.special import expit

from . import _json
from ._meta import requires
from .ontology import Ontology

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

_T = typing.TypeVar("_T", bound="ChemicalOntologyPredictor")

class ChemicalOntologyPredictor:
    """A model for predicting chemical hierarchy from BGC compositions.
    """

    def __init__(
        self,
        ontology: Ontology,
        n_jobs: Optional[int] = None,
        max_iter: int = 100,
        model: str = "logistic",
        alpha: float = 1.0,
    ) -> None:
        self.n_jobs: Optional[int] = n_jobs
        self.max_iter: int = max_iter
        self.classes_: Optional[pandas.DataFrame] = None
        self.features_: Optional[pandas.DataFrame] = None
        self.coef_: Optional[numpy.ndarray[float]] = None
        self.intercept_: Optional[numpy.ndarray[float]] = None
        self.ontology: Ontology = ontology
        self.model: str = model
        self.alpha: float = 1.0

    def __getstate__(self) -> Dict[str, object]:
        return {
            "classes_": self.classes_,
            "features_": self.features_,
            "intercept_": list(self.intercept_),
            "ontology": self.ontology.__getstate__(),
            "coef_": scipy.sparse.csr_matrix(self.coef_),
            "model": self.model,
            "alpha": self.alpha,
        }

    def __setstate__(self, state: Dict[str, object]) -> None:
        self.classes_ = state["classes_"]
        self.features_ = state["features_"]
        self.intercept_ = numpy.asarray(state["intercept_"])
        self.coef_ = state["coef_"].toarray()
        self.ontology.__setstate__(state["ontology"])
        self.model = state["model"]
        self.alpha = state.get("alpha", 1.0)

    @requires("sklearn.multiclass")
    @requires("sklearn.linear_model")
    @requires("sklearn.preprocessing")
    def _fit_logistic(self, X, Y):
        # train model using scikit-learn
        model = sklearn.multiclass.OneVsRestClassifier(
            sklearn.linear_model.LogisticRegression(
                "l1",
                solver="liblinear",
                max_iter=self.max_iter,
                C=1.0/self.alpha
            ),
            n_jobs=self.n_jobs,
        )
        model.fit(X, Y)

        # copy coefficients & intercept to a single NumPy array
        self.coef_ = numpy.zeros((X.shape[1], Y.shape[1]), order="C")
        self.intercept_ = numpy.zeros(Y.shape[1], order="C")
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

    @requires("sklearn.linear_model")
    def _fit_ridge(self, X, Y):
        # train model using scikit-learn
        model = sklearn.linear_model.RidgeClassifier(alpha=self.alpha)
        model.fit(X, Y)

        # copy coefficients & intercept to a single NumPy array
        self.coef_ = model.coef_.T
        self.intercept_ = model.intercept_

        # remove features with all-zero weights
        nonzero_weights = numpy.abs(self.coef_).sum(axis=1) > 0
        self.coef_ = self.coef_[nonzero_weights]
        self.features_ = self.features_[nonzero_weights]

    def fit(
        self: _T,
        X: Union[numpy.ndarray, anndata.AnnData],
        Y: Union[numpy.ndarray, anndata.AnnData],
    ) -> _T:
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

        # check training data consistency
        if _Y.shape[1] != len(self.ontology.incidence_matrix):
            raise ValueError(
                f"Ontology contains {len(self.ontology.incidence_matrix)} terms, "
                f"{_Y.shape[1]} found in data"
            )

        if self.model == "logistic":
            self._fit_logistic(_X, _Y)
        elif self.model == "ridge":
            self._fit_ridge(_X, _Y)

        return self

    def propagate(self, Y: numpy.ndarray) -> numpy.ndarray:
        _Y = numpy.array(Y, dtype=bool)
        for i in reversed(self.ontology.incidence_matrix):
            for j in self.ontology.incidence_matrix.parents(i):
                _Y[:, j] |= _Y[:, i]
        return _Y

    def _predict_logistic(self, X: numpy.ndarray) -> numpy.ndarray:
        return expit(X @ self.coef_ + self.intercept_)

    def _predict_ridge(self, X: numpy.ndarray) -> numpy.ndarray:
        result = X @ self.coef_ + self.intercept_
        probas = (result + 1.0) / 2.0
        return numpy.clip(probas, 0.0, 1.0)

    def predict_probas(
        self,
        X: Union[numpy.ndarray, anndata.AnnData],
    ) -> numpy.ndarray:
        if isinstance(X, anndata.AnnData):
            _X = X.X.toarray()
        else:
            _X = numpy.asarray(X)
        if self.model == "logistic":
            return self._predict_logistic(_X)
        elif self.model == "ridge":
            return self._predict_ridge(_X)

    def predict(
        self,
        X: Union[numpy.ndarray, anndata.AnnData],
        propagate: bool = True,
    ) -> numpy.ndarray:
        probas = self.predict_probas(X)
        classes = probas > 0.5
        if propagate:
            return self.propagate(classes)
        return classes

    def save(self, file: TextIO) -> None:
        state = self.__getstate__()
        json.dump(state, file, cls=_json.JSONEncoder, sort_keys=True, indent=1)

    @classmethod
    def trained(cls: Type[_T]) -> _T:
        with files(__package__).joinpath("predictor.json").open() as f:
            return cls.load(f)

    @classmethod
    def load(cls: Type[_T], file: TextIO) -> _T:
        state = json.load(file, cls=_json.JSONDecoder)
        predictor = cls(Ontology(None))
        predictor.__setstate__(state)
        return predictor

    def checksum(self, hasher: Optional[Any] = None) -> str:
        if hasher is None:
            hasher = hashlib.sha256()
        hasher.update(self.coef_)
        hasher.update(self.intercept_)
        return hasher.hexdigest()
