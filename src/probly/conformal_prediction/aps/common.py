"""Common functions for APS (Adaptive Prediction Sets) module.

Contains methods for calculating quantiles and non-conformity scores.
"""

from __future__ import annotations

import numpy as np


def calculate_quantile(scores: np.ndarray, alpha: float) -> float:
    """Calculate the quantile for conformal prediction.

    Parameters:
    scores : np.ndarray
            Non-conformity scores
    alpha : float
            Significance level (target coverage is 1-alpha)

    Returns:
    float (The (1-alpha)-quantile of the scores)
    """
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)  # ensure within [0, 1]
    return float(np.quantile(scores, q_level, method="lower"))


def calculate_nonconformity_score(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Calculate APS non-conformity scores for given true labels.

    Parameters:
    probabilities : np.ndarray
        Predicted probabilities of shape (n_samples, n_classes)
    labels : np.ndarray
        True labels of shape (n_samples)

    Returns:
    np.ndarray
        Non-conformity scores of shape (n_samples)
    """
    n_samples = probabilities.shape[0]
    scores = np.zeros(n_samples)

    for i in range(n_samples):
        probs = probabilities[i]
        sorted_items = sorted([(-probs[j], j) for j in range(len(probs))])
        # Get descending sorted probabilities
        sorted_indices = [idx for (_, idx) in sorted_items]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)

        # find pos of true label in sorted order
        # will fail if labels[i] is out of bounds, but is expected
        true_label_pos = sorted_indices.index(labels[i])
        scores[i] = cumulative_probs[true_label_pos].item()

    return scores
