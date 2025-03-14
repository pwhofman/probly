import numpy as np
from scipy.stats import entropy
from scipy.optimize import minimize

def total_uncertainty_entropy(probs, base=2):
    """
    Compute the total uncertainty using samples from a second-order distribution.
    Args:
    probs: numpy.ndarray of shape (n_instances, n_samples, n_classes)
    base: int, default=2

    Returns:
    tu: numpy.ndarray of shape (n_instances,)
    """
    tu = entropy(probs.mean(axis=1), axis=1, base=base)
    return tu

def aleatoric_uncertainty_entropy(probs, base=2):
    """
        Compute the aleatoric uncertainty using samples from a second-order distribution.
        Args:
        probs: numpy.ndarray of shape (n_instances, n_samples, n_classes)
        base: int, default=2

        Returns:
        au: numpy.ndarray of shape (n_instances,)
    """
    au = entropy(probs, axis=2, base=base).mean(axis=1)
    return au

def epistemic_uncertainty_entropy(probs, base=2):
    """
        Compute the epistemic uncertainty using samples from a second-order distribution.
        Args:
        probs: numpy.ndarray of shape (n_instances, n_samples, n_classes)
        base: int, default=2

        Returns:
        eu: numpy.ndarray of shape (n_instances,)
    """
    probs_mean = probs.mean(axis=1)
    probs_mean = np.repeat(np.expand_dims(probs_mean, 1), repeats=probs.shape[1], axis=1)
    eu = entropy(probs, probs_mean, axis=2, base=base).mean(axis=1)
    return eu

def total_uncertainty_loss(probs, loss):
    mean = np.mean(probs, axis=1)
    tu = np.sum(mean * loss(mean), axis=1)
    return tu

def aleatoric_uncertainty_loss(probs, loss):
    au = np.mean(np.sum(probs * loss(probs), axis=2), axis=1)
    return au

def epistemic_uncertainty_loss(probs, loss):
    mean = np.mean(probs, axis=1)
    eu = (np.sum(mean * loss(mean), axis=1) -
          np.mean(np.sum(probs * loss(probs), axis=2), axis=1))
    return eu

def total_uncertainty_variance(probs):
    probs_mean = probs.mean(axis=1)
    tu = np.sum(probs_mean * (1 - probs_mean), axis=1)
    return tu

def aleatoric_uncertainty_variance(probs):
    au = np.sum(np.mean(probs * (1 - probs), axis=1), axis=1)
    return au

def epistemic_uncertainty_variance(probs):
    probs_mean = probs.mean(axis=1, keepdims=True)
    eu = np.sum(np.mean(probs * (probs - probs_mean), axis=1), axis=1)
    return eu

def total_uncertainty_distance(probs):
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

def aleatoric_uncertainty_distance(probs):
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

def epistemic_uncertainty_distance(probs):
    """
    Compute the epistemic uncertainty using samples from a second-order distribution.
    The measure of epistemic uncertainty is from https://arxiv.org/pdf/2312.00995.
    Args:
        probs: numpy.ndarray of shape (n_instances, n_samples, n_classes)
    Returns:
        tu: numpy.ndarray of shape (n_instances,)
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
