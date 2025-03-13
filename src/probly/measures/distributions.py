import numpy as np
from scipy.stats import entropy

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
