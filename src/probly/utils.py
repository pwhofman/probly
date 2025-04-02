"""General utility functions for all other modules."""

import itertools
from collections.abc import Iterable

import numpy as np
import torch


def powerset(iterable: Iterable) -> list[tuple]:
    """Generate the power set of a given iterable.

    Args:
        iterable: Iterable
    Returns:
        List[tuple], power set of the given iterable

    """
    s = list(iterable)
    return list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1)))


def capacity(q: np.ndarray, a: Iterable) -> np.ndarray:
    """Compute the capacity of set q given set a.

    Args:
        q: numpy.ndarray, shape (n_instances, n_samples, n_classes)
        a: Iterable, shape (n_classes,), indices indicating subset of classes
    Returns:
        min_capacity: numpy.ndarray, shape (n_instances,), capacity of q given a

    """
    selected_sum = np.sum(q[:, :, a], axis=2)
    min_capacity = np.min(selected_sum, axis=1)
    return min_capacity


def moebius(q: np.ndarray, a: Iterable) -> np.ndarray:
    """Compute the Moebius function of a set q given a set a.

    Args:
        q: numpy.ndarray of shape (num_samples, num_members, num_classes)
        a: numpy.ndarray, shape (n_classes,), indices indicating subset of classes
    Returns:
        m_a: numpy.ndarray, shape (n_instances,), moebius value of q given a

    """
    ps_a = powerset(a)  # powerset of A
    ps_a.pop(0)  # remove empty set
    m_a = np.zeros(q.shape[0])
    for b in ps_a:
        dl = len(set(a) - set(b))
        m_a += ((-1) ** dl) * capacity(q, b)
    return m_a


def differential_entropy_gaussian(sigma2: float | np.ndarray, base: float = 2) -> float | np.ndarray:
    """Compute the differential entropy of a Gaussian distribution given the variance.

    https://en.wikipedia.org/wiki/Differential_entropy
    Args:
        sigma2: float or numpy.ndarray shape (n_instances,), variance of the Gaussian distribution
        base: float, base of the logarithm
    Returns:
        diff_ent: float or numpy.ndarray shape (n_instances,), differential entropy of the Gaussian distribution
    """
    return 0.5 * np.log(2 * np.pi * np.e * sigma2) / np.log(base)


def kl_divergence_gaussian(
    mu1: float | np.ndarray,
    sigma21: float | np.ndarray,
    mu2: float | np.ndarray,
    sigma22: float | np.ndarray,
    base: float = 2,
) -> float | np.ndarray:
    """Compute the KL-divergence between two Gaussian distributions.

    https://en.wikipedia.org/wiki/Kullback-Leibler_divergence#Examples
    Args:
        mu1: float or numpy.ndarray shape (n_instances,), mean of the first Gaussian distribution
        sigma21: float or numpy.ndarray shape (n_instances,), variance of the first Gaussian distribution
        mu2: float or numpy.ndarray shape (n_instances,), mean of the second Gaussian distribution
        sigma22: float or numpy.ndarray shape (n_instances,), variance of the second Gaussian distribution
        base: float, base of the logarithm
    Returns:
        kl_div: float or numpy.ndarray shape (n_instances,), KL-divergence between the two Gaussian distributions
    """
    kl_div = 0.5 * np.log(sigma22 / sigma21) / np.log(base) + (sigma21 + (mu1 - mu2) ** 2) / (2 * sigma22) - 0.5
    return kl_div


def torch_reset_all_parameters(module: torch.nn.Module) -> None:
    """Reset all parameters of a torch module.

    Args:
        module: torch.nn.Module, module to reset parameters

    """
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    for child in module.children():
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()
