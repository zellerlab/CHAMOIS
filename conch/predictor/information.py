"""Information-theoric evaluation metrics for ontological predictions.

References:
    Clark WT, Radivojac P. *Information-theoretic evaluation of predicted 
    ontological annotations*. Bioinformatics. 2013;29(13):i53-i61. 
    :doi:`10.1093/bioinformatics/btt228`.

"""
import math
from typing import Union

import numpy
import numpy.ma
import scipy.sparse
from sklearn.metrics import roc_curve

from ..ontology import IncidenceMatrix


def information_accretion(
    y_true: Union[numpy.ndarray, scipy.sparse.spmatrix], 
    hierarchy: IncidenceMatrix
) -> numpy.ndarray:
    """Compute the information accretion using frequencies from the given labels.
    """
    _Y = y_true.toarray() if isinstance(y_true, scipy.sparse.spmatrix) else y_true
    freq = numpy.zeros(_Y.shape[1], dtype=numpy.float32)
    for i in hierarchy:
        parents = hierarchy.parents(i)
        if parents.shape[0] > 0:
            all_parents = _Y[:, parents].all(axis=1)
            pos = _Y[all_parents, i].sum()
            tot = all_parents.sum()
        else:
            pos = _Y[:, i].sum()
            tot = _Y.shape[0]
        if tot > 0 and pos > 0:
            freq[i] = pos / tot
    return -numpy.log2(freq, where=freq != 0.0)


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
    ru[:] = ia.sum(axis=0, where=not_predicted)
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
    mi[:] = ia.sum(axis=0, where=wrong_prediction)
    return mi


def semantic_distance_score(y_true, y_scores, information_accretion, *, k=2):
    ru, mi, _ = information_theoric_curve(y_true, y_scores, information_accretion)
    return (ru**k + mi**k).min()**(1 / k)


def information_theoric_curve(y_true, y_scores, information_accretion):
    """Return the information theoric curve for the predictions.
    """
    _, _, thresholds = roc_curve(y_true.ravel(), y_scores.ravel(), drop_intermediate=True)

    mi = numpy.zeros_like(thresholds)
    ru = numpy.zeros_like(thresholds)

    ia = numpy.tile(information_accretion, y_true.shape[0]).reshape(y_true.shape)

    for i, t in enumerate(thresholds):
        y_pred = y_scores >= t
        fp = y_pred & (~y_true)
        fn = y_true & (~y_pred)
        mi[i] = ia.sum(where=fp)
        ru[i] = ia.sum(where=fn)

    mi /= y_true.shape[0]
    ru /= y_pred.shape[0]
    return mi, ru, thresholds


def weighted_information_theoric_curve(y_true, y_scores, information_accretion):
    """Return the weighted information theoric curve for the predictions.
    """
    # scores = numpy.sort(numpy.unique(y_scores.ravel()))
    # n = 1 if len(scores) < 50 else len(scores) // 50
    # thresholds = scores[::n]
    _, _, thresholds = roc_curve(y_true.ravel(), y_scores.ravel(), drop_intermediate=True)

    mi = numpy.zeros_like(thresholds)
    ru = numpy.zeros_like(thresholds)

    ia = numpy.tile(information_accretion, y_true.shape[0]).reshape(y_true.shape)
    ic = ia.sum(axis=1, where=y_true)
    ia_w = ia * ic

    for i, t in enumerate(thresholds):
        y_pred = y_scores >= t
        fp = y_pred & (~y_true)
        fn = y_true & (~y_pred)
        # mi[i] = (ia.sum(axis=1, where=fp)*ic).sum()
        # ru[i] = (ia.sum(axis=1, where=fn)*ic).sum()
        mi[i] = ia_w.sum(where=fp)
        ru[i] = ia_w.sum(where=fn)

    mi /= ic.sum()
    ru /= ic.sum()
    return mi, ru, thresholds


def weighted_precision_recall_curve(y_true, y_scores, information_accretion):
    # scores = numpy.sort(numpy.unique(y_scores.ravel()))
    # n = 1 if len(scores) < 50 else len(scores) // 50
    # thresholds = scores[::n]
    _, _, thresholds = roc_curve(y_true.ravel(), y_scores.ravel(), drop_intermediate=True)

    pr = numpy.zeros_like(thresholds)
    rc = numpy.zeros_like(thresholds)
   
    ia = numpy.tile(information_accretion, y_true.shape[0]).reshape(y_true.shape)
    ic = ia.sum(axis=1, where=y_true)

    for i, t in enumerate(thresholds):
        y_pred = y_scores >= t
        tp = y_pred & y_true
        n = ia.sum(axis=1, where=tp)
        pr[i] = ( numpy.nan_to_num(n / ia.sum(axis=1, where=y_pred), 1) * ic).sum()
        rc[i] = ( numpy.nan_to_num(n / ic, 0) * ic).sum()

    pr /= ic.sum()
    rc /= ic.sum()
    i = rc.argmin()
    pr[i] = 1.0
    rc[i] = 0.0
    return pr, rc, thresholds
