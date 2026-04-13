"""General utility functions for all other modules."""

from __future__ import annotations

import numpy as np


def differential_entropy_gaussian(sigma2: float | np.ndarray, base: float = 2) -> float | np.ndarray:
    """Compute the differential entropy of a Gaussian distribution given the variance.

    Args:
        sigma2: Variance of the Gaussian distribution.
        base: Base of the logarithm.

    Returns:
        Differential entropy of the Gaussian distribution.

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

    Args:
        mu1: Mean of the first Gaussian distribution.
        sigma21: Variance of the first Gaussian distribution.
        mu2: Mean of the second Gaussian distribution.
        sigma22: Variance of the second Gaussian distribution.
        base: Base of the logarithm.

    Returns:
        KL-divergence between the two Gaussian distributions.

    """
    kl_div = 0.5 * np.log(sigma22 / sigma21) / np.log(base) + (sigma21 + (mu1 - mu2) ** 2) / (2 * sigma22) - 0.5
    return kl_div


def intersection_probability(probs: np.ndarray) -> np.ndarray:
    """Compute the intersection probability of a credal set.

    Computes the intersection probability from :cite:`cuzzolinIntersectionProbability2022`.

    Args:
        probs: Credal sets of shape (n_instances, n_samples, n_classes).

    Returns:
        Intersection probability of the credal sets of shape (n_instances, n_classes).

    """
    lower = np.min(probs, axis=1)
    upper = np.max(probs, axis=1)
    diff = upper - lower
    diff_sum = np.sum(diff, axis=1)
    lower_sum = np.sum(lower, axis=1)
    # Compute alpha for instances for which probability intervals are not empty, otherwise set alpha to 0.
    alpha = np.zeros(probs.shape[0])
    nonzero_idxs = diff_sum != 0
    alpha[nonzero_idxs] = (1 - lower_sum[nonzero_idxs]) / diff_sum[nonzero_idxs]
    int_probs = lower + alpha[:, None] * diff
    return int_probs
