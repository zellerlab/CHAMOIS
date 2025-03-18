import contextlib
import hashlib
import json
import math
import pickle
import typing
import importlib.resources
from typing import Any, Literal, List, Tuple, Dict, TextIO, Type, Union, Optional, Callable

import numpy

from .. import _json
from .._meta import requires
from ..ontology import Ontology
from .information import information_accretion

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

if typing.TYPE_CHECKING:
    from anndata import AnnData
    from scipy.sparse import spmatrix
    from pandas import DataFrame


_T = typing.TypeVar("_T", bound="ChemicalOntologyPredictor")

class ChemicalOntologyPredictor:
    """A model for predicting chemical hierarchy from BGC compositions.
    """

    _MODELS = ["ridge", "logistic", "dummy"]

    def __init__(
        self,
        ontology: Ontology,
        n_jobs: Optional[int] = None,
        max_iter: int = 100,
        model: str = "logistic",
        alpha: float = 1.0,
        variance: Optional[float] = None,
    ) -> None:
        if model not in self._MODELS:
            raise ValueError(f"invalid model architecture: {model!r}")
        self.n_jobs: Optional[int] = n_jobs
        self.max_iter: int = max_iter
        self.classes_: Optional["DataFrame"] = None
        self.features_: Optional["DataFrame"] = None
        self.coef_: Optional[numpy.ndarray[float]] = None
        self.intercept_: Optional[numpy.ndarray[float]] = None
        self.ontology: Ontology = ontology
        self.model: str = model
        self.alpha: float = 1.0
        self.variance = variance

    def __getstate__(self) -> Dict[str, object]:
        return {
            "classes_": self.classes_,
            "features_": self.features_,
            "intercept_": list(self.intercept_),
            "ontology": self.ontology.__getstate__(),
            "coef_": self.coef_,
            "model": self.model,
            "alpha": self.alpha,
            "variance": self.variance
        }

    def __setstate__(self, state: Dict[str, object]) -> None:
        self.classes_ = state["classes_"]
        self.features_ = state["features_"]
        self.intercept_ = numpy.asarray(state["intercept_"])
        self.coef_ = state["coef_"]
        self.ontology.__setstate__(state["ontology"])
        self.model = state["model"]
        self.alpha = state.get("alpha", 1.0)
        self.variance = state.get("variance", None)

    @requires("sklearn.feature_selection")
    def _select_features(self, X: Union[numpy.ndarray, "spmatrix"]):
        import scipy.sparse

        _X = X.toarray() if isinstance(X, scipy.sparse.spmatrix) else X
        varfilt = sklearn.feature_selection.VarianceThreshold(self.variance)
        varfilt.fit(_X)
        support = varfilt.get_support()
        self.features_ = self.features_.loc[support]
        return _X[:, support]

    def _compute_information_accretion(self, Y: Union[numpy.ndarray, "spmatrix"]):
        import scipy.sparse

        _Y = Y.toarray() if isinstance(Y, scipy.sparse.spmatrix) else Y
        ia = numpy.zeros(Y.shape[1])
        for i in self.ontology.adjacency_matrix:
            parents = self.ontology.adjacency_matrix.parents(i)
            assert parents.shape[0] <= 1

            if len(parents) == 1:
                pos = _Y[_Y[:, parents[0]], i].sum()
                tot = _Y[:, parents[0]].sum()
                if tot > 0.0 and pos > 0.0:
                    freq = pos / tot
                    ia[i] = - math.log2(freq)
        self.classes_["information_accretion"] = ia

    @requires("sklearn.multiclass")
    @requires("sklearn.linear_model")
    @requires("sklearn.preprocessing")
    @requires("scipy.sparse")
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
    
        # store weights in sparse matrix
        self.coef_ = scipy.sparse.csr_matrix(self.coef_)

    @requires("sklearn.linear_model")
    @requires("scipy.sparse")
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

        # store weights in sparse matrix
        self.coef_ = scipy.sparse.csr_matrix(self.coef_)

    @requires("scipy.sparse")
    @requires("scipy.special")
    def _fit_dummy(self, X, Y):
        self.intercept_ = numpy.zeros(Y.shape[1])
        self.coef_ = numpy.zeros((0, Y.shape[1]))
        for i in range(Y.shape[1]):           
            n_pos = Y[:, i].sum()
            odds = scipy.special.logit(n_pos / Y.shape[0])
            self.intercept_[i] = numpy.clip(odds, -10, 10)
        self.coef_ = scipy.sparse.csr_matrix(self.coef_)

    @requires("pandas")
    def fit(
        self: _T,
        X: Union[numpy.ndarray, "AnnData"],
        Y: Union[numpy.ndarray, "AnnData"],
    ) -> _T:
        import anndata

        if isinstance(X, anndata.AnnData):
            _X = X.X
            self.features_ = X.var.copy()
        else:
            _X = X
            self.features_ = pandas.DataFrame(index=list(map(str, range(1, _X.shape[1] + 1))))
        if isinstance(Y, anndata.AnnData):
            _Y = Y.X
            self.classes_ = Y.var.copy()
        else:
            _Y = Y
            self.classes_ = pandas.DataFrame(index=list(map(str, range(1, _Y.shape[1] + 1))))

        # check training data consistency
        if _Y.shape[1] != len(self.ontology.adjacency_matrix):
            raise ValueError(
                f"Ontology contains {len(self.ontology.adjacency_matrix)} terms, "
                f"{_Y.shape[1]} found in data"
            )
        # compute information accretion
        self.classes_["information_accretion"] = information_accretion(_Y, self.ontology.adjacency_matrix)
        self._compute_information_accretion(_Y)
        # run variance selection if requested
        if self.variance is not None:
            _X = self._select_features(_X)

        if self.model == "logistic":
            self._fit_logistic(_X, _Y)
        elif self.model == "ridge":
            self._fit_ridge(_X, _Y)
        elif self.model == "dummy":
            self._fit_dummy(_X, _Y)

        return self

    def propagate(self, Y: numpy.ndarray) -> numpy.ndarray:
        assert Y.shape[1] == len(self.ontology.adjacency_matrix)
        _Y = numpy.array(Y, dtype=Y.dtype)
        for i in reversed(self.ontology.adjacency_matrix):
            for j in self.ontology.adjacency_matrix.parents(i):
                _Y[:, j] = numpy.maximum(_Y[:, j], _Y[:, i])
        return _Y

    @requires("scipy.special")
    def _predict_logistic(self, X: numpy.ndarray) -> numpy.ndarray:
        return scipy.special.expit(X @ self.coef_ + self.intercept_)

    def _predict_ridge(self, X: numpy.ndarray) -> numpy.ndarray:
        result = X @ self.coef_ + self.intercept_
        probas = (result + 1.0) / 2.0
        return numpy.clip(probas, 0.0, 1.0)

    @requires("scipy.special")
    def _predict_dummy(self, X: numpy.ndarray) -> numpy.ndarray:
        y = scipy.special.expit(self.intercept_)
        return numpy.tile(y, (X.shape[0], 1))

    def predict_probas(
        self,
        X: Union[numpy.ndarray, "AnnData"],
        propagate: bool = True,
    ) -> numpy.ndarray:
        import anndata

        if isinstance(X, anndata.AnnData):
            _X = X.X
        else:
            _X = numpy.asarray(X)
        if self.model == "logistic":
            probas = self._predict_logistic(_X)
        elif self.model == "ridge":
            probas = self._predict_ridge(_X)
        elif self.model == "dummy":
            probas = self._predict_dummy(_X)
        else:
            raise RuntimeError(f"invalid model architecture: {self.model!r}")
        if propagate:
            probas = self.propagate(probas)
        return probas

    def predict(
        self,
        X: Union[numpy.ndarray, "AnnData"],
        propagate: bool = True,
    ) -> numpy.ndarray:
        probas = self.predict_probas(X)
        classes = probas > 0.5
        if propagate:
            return self.propagate(classes)
        return classes

    def information_content(self, Y: numpy.ndarray) -> numpy.ndarray:
        r"""Compute the information content of a prediction.

        The information content for an annotation subgraph :math:`ic(T)` is
        defined as the sum of information accretion :math:`ia(i)` for every
        node :math:`i` of the subgraph :math:`T`. Information accretion is
        computed from partial probabilities extracted from the training set:

        .. math::

            ia(T) = \sum_{i \in T}{ - log_2(P(i | \mathcal{P}(i))) }

        where :math:`\mathcal{P}(i)` is the parent of node :math:`i` in the
        ontology graph.

        Arguments:
            Y (`numpy.ndarray` of shape (n_samples, n_classes)): The array
                of predicted class labels for which to compute

        Returns:
            `numpy.ndarray` of shape (n_samples,): The computed information
            content for each sample prediction.

        References:
            Clark WT, Radivojac P. *Information-theoretic evaluation of
            predicted ontological annotations*. Bioinformatics.
            2013;29(13):i53-i61. :doi:`10.1093/bioinformatics/btt228`.

        """
        if Y.dtype != numpy.bool_:
            raise TypeError(f"Expected bool matrix, found {Y.dtype}")
        ic = numpy.zeros(Y.shape[0])
        for i in range(Y.shape[0]):
            ic[i] = self.classes_["information_accretion"][Y[i]].sum()
        return ic

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
        hasher.update(self.coef_.toarray())
        hasher.update(self.intercept_)
        return hasher.hexdigest()
