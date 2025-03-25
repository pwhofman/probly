from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy

from ..utils import moebius, powerset


def total_entropy(probs: np.ndarray, base: int = 2) -> np.ndarray:
    """
    Compute the total uncertainty using samples from a second-order distribution.
    Args:
        probs: numpy.ndarray of shape (n_instances, n_samples, n_classes)
        base: int, default=2
    Returns:
        te: numpy.ndarray of shape (n_instances,)
    """
    te = entropy(probs.mean(axis=1), axis=1, base=base)
    return te


def conditional_entropy(probs: np.ndarray, base: int = 2) -> np.ndarray:
    """
    Compute the aleatoric uncertainty using samples from a second-order distribution.
    Args:
        probs: numpy.ndarray of shape (n_instances, n_samples, n_classes)
        base: int, default=2
    Returns:
        ce: numpy.ndarray of shape (n_instances,)
    """
    ce = entropy(probs, axis=2, base=base).mean(axis=1)
    return ce


def mutual_information(probs: np.ndarray, base: int = 2) -> np.ndarray:
    """
    Compute the epistemic uncertainty using samples from a second-order distribution.
    Args:
        probs: numpy.ndarray of shape (n_instances, n_samples, n_classes)
        base: int, default=2
    Returns:
        mi: numpy.ndarray of shape (n_instances,)
    """
    probs_mean = probs.mean(axis=1)
    probs_mean = np.repeat(np.expand_dims(probs_mean, 1), repeats=probs.shape[1], axis=1)
    mi = entropy(probs, probs_mean, axis=2, base=base).mean(axis=1)
    return mi


def expected_loss(probs: np.ndarray,
                  loss_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Computes the expected loss of the second-order distribution using samples
    from the second-order distribution.
    Args:
        probs: numpy.ndarray of shape (n_instances, n_samples, n_classes)
        loss_fn: Callable[[numpy.ndarray], numpy.ndarray]
    Returns:
        el: numpy.ndarray, shape (n_instances,)
    """
    mean = np.mean(probs, axis=1)
    el = np.sum(mean * loss_fn(mean), axis=1)
    return el


def expected_entropy(probs: np.ndarray,
                     loss_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Computes the expected entropy of the second-order distribution using samples
    from the second-order distribution.
    Args:
        probs: numpy.ndarray of shape (n_instances, n_samples, n_classes)
        loss_fn: Callable[[numpy.ndarray], numpy.ndarray]
    Returns:
        ee: numpy.ndarray, shape (n_instances,)
    """
    ee = np.mean(np.sum(probs * loss_fn(probs), axis=2), axis=1)
    return ee


def expected_divergence(probs: np.ndarray,
                        loss_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Computes the expected divergence to the mean of the second-order distribution using samples
    from the second-order distribution.
    Args:
        probs: numpy.ndarray of shape (n_instances, n_samples, n_classes)
        loss_fn: Callable[[numpy.ndarray], numpy.ndarray]
    Returns:
        ed: numpy.ndarray, shape (n_instances,)
    """
    mean = np.mean(probs, axis=1)
    ed = (np.sum(mean * loss_fn(mean), axis=1) -
          np.mean(np.sum(probs * loss_fn(probs), axis=2), axis=1))
    return ed


def total_variance(probs: np.ndarray) -> np.ndarray:
    """
    Computes the total uncertainty using variance-based measures based on samples from
    a second-order distribution.
    Args:
        probs: numpy.ndarray of shape (n_instances, n_samples, n_classes)
    Returns:
        tv: numpy.ndarray, shape (n_instances,)
    """
    probs_mean = probs.mean(axis=1)
    tv = np.sum(probs_mean * (1 - probs_mean), axis=1)
    return tv


def expected_conditional_variance(probs: np.ndarray) -> np.ndarray:
    """
    Computes the aleatoric uncertainty using variance-based measures based on samples from
    a second-order distribution.
    Args:
        probs: numpy.ndarray of shape (n_instances, n_samples, n_classes)
    Returns:
        ecv: numpy.ndarray, shape (n_instances,)
    """
    ecv = np.sum(np.mean(probs * (1 - probs), axis=1), axis=1)
    return ecv


def variance_conditional_expectation(probs: np.ndarray) -> np.ndarray:
    """
    Computes the epistemic uncertainty using variance-based measures based on samples from
    a second-order distribution.
    Args:
        probs: numpy.ndarray of shape (n_instances, n_samples, n_classes)
    Returns:
        ecv: numpy.ndarray, shape (n_instances,)
    """
    probs_mean = probs.mean(axis=1, keepdims=True)
    vce = np.sum(np.mean(probs * (probs - probs_mean), axis=1), axis=1)
    return vce


def total_uncertainty_distance(probs: np.ndarray) -> np.ndarray:
    """
    Compute the total uncertainty using samples from a second-order distribution.
    The measure of total uncertainty is from https://arxiv.org/pdf/2312.00995.
    Args:
        probs: numpy.ndarray of shape (n_instances, n_samples, n_classes)
    Returns:
        tu: numpy.ndarray of shape (n_instances,)
    """
    probs_mean = probs.mean(axis=1)
    tu = 1 - np.max(probs_mean, axis=1)
    return tu


def aleatoric_uncertainty_distance(probs: np.ndarray) -> np.ndarray:
    """
    Compute the aleatoric uncertainty using samples from a second-order distribution.
    The measure of aleatoric uncertainty is from https://arxiv.org/pdf/2312.00995.
    Args:
        probs: numpy.ndarray of shape (n_instances, n_samples, n_classes)
    Returns:
        tu: numpy.ndarray of shape (n_instances,)
    """
    au = 1 - np.mean(np.max(probs, axis=2), axis=1)
    return au


def epistemic_uncertainty_distance(probs: np.ndarray) -> np.ndarray:
    """
    Compute the epistemic uncertainty using samples from a second-order distribution.
    The measure of epistemic uncertainty is from https://arxiv.org/pdf/2312.00995.
    Args:
        probs: numpy.ndarray of shape (n_instances, n_samples, n_classes)
    Returns:
        eu: numpy.ndarray of shape (n_instances,)
    """

    def fun(q, p):
        f = np.mean(np.linalg.norm(p - q[None, :], ord=1, axis=1))
        return f

    x0 = probs.mean(axis=1)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = [(0, 1)] * probs.shape[2]
    eu = np.empty(probs.shape[0])
    for i in range(probs.shape[0]):
        res = minimize(fun=fun, x0=x0[i],
                       bounds=bounds,
                       constraints=constraints,
                       args=probs[i])
        eu[i] = 0.5 * res.fun
    return eu


def upper_entropy(probs: np.ndarray, base: int = 2) -> np.ndarray:
    """
    Computes the upper entropy of a credal set. Given the probs array the lower and upper
    probabilities are computed and the credal set is assumed to be a convex set including all
    probability distributions in the interval [lower, upper] for all classes. The upper entropy
    of this set is computed.
    Args:
        probs: numpy.ndarray of shape (n_instances, n_samples, n_classes)
        base: int, default=2
    Returns:
        ue: numpy.ndarray of shape (n_instances,)
    """

    def fun(x):
        return -entropy(x, base=base)

    x0 = probs.mean(axis=1)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    ue = np.empty(probs.shape[0])
    for i in range(probs.shape[0]):
        bounds = list(zip(np.min(probs[i], axis=0), np.max(probs[i], axis=0)))
        res = minimize(fun=fun, x0=x0[i], bounds=bounds, constraints=constraints)
        ue[i] = -res.fun
    return ue


def lower_entropy(probs: np.ndarray, base: int = 2) -> np.ndarray:
    """
     Computes the lower entropy of a credal set. Given the probs array the lower and upper
     probabilities are computed and the credal set is assumed to be a convex set including all
     probability distributions in the interval [lower, upper] for all classes. The lower entropy
     of this set is computed.
     Args:
         probs: numpy.ndarray of shape (n_instances, n_samples, n_classes)
         base: int, default=2
     Returns:
         le: numpy.ndarray of shape (n_instances,)
     """

    def fun(x):
        return entropy(x, base=base)

    x0 = probs.mean(axis=1)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    le = np.empty(probs.shape[0])
    for i in range(probs.shape[0]):
        bounds = list(zip(np.min(probs[i], axis=0), np.max(probs[i], axis=0)))
        res = minimize(fun=fun, x0=x0[i], bounds=bounds, constraints=constraints)
        le[i] = -res.fun
    return le


def generalised_hartley(probs: np.ndarray, base: int = 2) -> np.ndarray:
    """
    Computes the generalised Hartley measure given the extreme points of
    a credal set.
    Args:
        probs: numpy.ndarray of shape (n_instances, n_samples, n_classes)
        base: int, default=2
    Returns:
        gh: numpy.ndarray of shape (n_instances,)
    """
    gh = np.zeros(probs.shape[0])
    idxs = list(range(probs.shape[2]))  # list of class indices
    ps_A = powerset(idxs)  # powerset of all indices
    ps_A.pop(0)  # remove empty set
    for A in ps_A:
        m_A = moebius(probs, A)
        gh += m_A * (np.log(len(A)) / np.log(base))
    return gh


def evidential_uncertainty(evidences: np.ndarray) -> np.ndarray:
    """
    Compute the evidential uncertainty given the evidences.
    Args:
        evidences: numpy.ndarray of shape (n_instances, n_classes)
    Returns:
        eu: numpy.ndarray of shape (n_instances,)
    """
    strenghts = np.sum(evidences + 1.0, axis=1)
    K = np.full(evidences.shape[0], evidences.shape[1])
    eu = K / strenghts
    return eu
