"""Information-theoric evaluation metrics for ontological predictions.

References:
    Clark WT, Radivojac P. *Information-theoretic evaluation of predicted 
    ontological annotations*. Bioinformatics. 2013;29(13):i53-i61. 
    :doi:`10.1093/bioinformatics/btt228`.

"""
import math
from typing import Union

import numpy
import scipy.sparse

from ..ontology import IncidenceMatrix


def information_accretion(
    y_true: Union[numpy.ndarray, scipy.sparse.spmatrix], 
    hierarchy: IncidenceMatrix
) -> numpy.ndarray:
    """Compute the information accretion using frequencies from the given labels.
    """
    _Y = y_true.toarray() if isinstance(y_true, scipy.sparse.spmatrix) else y_true
    ia = numpy.zeros(_Y.shape[1])
    for i in hierarchy:
        parents = hierarchy.parents(i)
        if parents.shape[0] > 0:
            all_parents = _Y[:, parents].all(axis=1)
            pos = _Y[all_parents, i].sum()
            tot = all_parents.sum()
            freq = pos / tot
            if freq > 0.0:
                ia[i] = - math.log2(freq)
    return ia


def remaining_uncertainty_score(y_true, y_pred, information_accretion):
    """Compute the remaining uncertainty score for a prediction.

    Arguments:
        y_true (`numpy.ndarray` of shape (n_samples, n_classes)): The true
            labels for all observations.
        y_pred (`numpy.ndarray` of shape (n_samples, n_classes)): The 
            predicted labels for all observations.
        information_accretion (`numpy.ndarray` of shape (n_classes,)): The
            information accretion for each class.

    """
    ru = numpy.zeros(y_true.shape[0])
    not_predicted = y_true & (~y_pred)
    ia = numpy.tile(information_accretion, y_true.shape[0])
    ru[:] = (ia * not_predicted).sum(axis=0)
    return ru


def misinformation_score(y_true, y_pred, information_accretion):
    """Compute the misinformation score for a prediction.

    Arguments:
        y_true (`numpy.ndarray` of shape (n_samples, n_classes)): The true
            labels for all observations.
        y_pred (`numpy.ndarray` of shape (n_samples, n_classes)): The 
            predicted labels for all observations.
        information_accretion (`numpy.ndarray` of shape (n_classes,)): The
            information accretion for each class.

    """
    mi = numpy.zeros(y_true.shape[0])
    wrong_prediction = y_pred & (~y_true)
    ia = numpy.tile(information_accretion, y_true.shape[0]).reshape(y_true.shape)
    mi[:] = (ia * wrong_prediction).sum(axis=0)
    return mi


def semantic_distance_score(y_true, y_scores, information_accretion, *, k=2):
    ru, mi, _ = information_theoric_curve(y_true, y_scores, information_accretion)
    return (ru**k + mi**k).min()**(1 / k)


def information_theoric_curve(y_true, y_scores, information_accretion):
    """Return the information theoric curve for the predictions.
    """
    scores = numpy.sort(numpy.unique(y_scores.ravel()))
    thresholds = scores[::len(scores)//50]

    mi = numpy.zeros_like(thresholds)
    ru = numpy.zeros_like(thresholds)
    ia = numpy.tile(information_accretion, y_true.shape[0]).reshape(y_true.shape)

    for i, t in enumerate(thresholds):
        y_pred = y_scores >= t
        fp = y_pred & (~y_true)
        fn = y_true & (~y_pred)
        mi[i] = (ia * fp).sum()
        ru[i] = (ia * fn).sum()

    mi /= y_true.shape[0]
    ru /= y_pred.shape[0]
    return mi, ru, thresholds


def weighted_information_theoric_curve(y_true, y_scores, information_accretion):
    """Return the weighted information theoric curve for the predictions.
    """
    scores = numpy.sort(numpy.unique(y_scores.ravel()))
    thresholds = scores[::len(scores)//50]

    mi = numpy.zeros_like(thresholds)
    ru = numpy.zeros_like(thresholds)
    ia = numpy.tile(information_accretion, y_true.shape[0]).reshape(y_true.shape)

    ic = (ia * y_true).sum(axis=1)
    assert ic.shape[0] == y_true.shape[0]

    for i, t in enumerate(thresholds):
        y_pred = y_scores >= t
        fp = y_pred & (~y_true)
        fn = y_true & (~y_pred)
        mi[i] = ((ia * fp).sum(axis=1) * ic).sum()
        ru[i] = ((ia * fn).sum(axis=1) * ic).sum()

    mi /= ic.sum()
    ru /= ic.sum()
    return mi, ru, thresholds