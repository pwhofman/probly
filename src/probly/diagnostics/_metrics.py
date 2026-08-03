"""Backend-agnostic metric helpers for diagnostics."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np


def to_numpy(values: Any) -> np.ndarray:  # noqa: ANN401
    """Convert an array-like (including torch tensors) to a numpy array."""
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    return np.asarray(values)


def area_under_risk_coverage(uncertainty: np.ndarray, errors: np.ndarray) -> float:
    """Area under the risk-coverage curve when rejecting by decreasing uncertainty."""
    order = np.argsort(uncertainty, kind="stable")
    risks = np.cumsum(errors[order]) / np.arange(1, len(errors) + 1)
    return float(risks.mean())


def expected_calibration_error(probabilities: np.ndarray, targets: np.ndarray, num_bins: int = 15) -> float:
    """Expected calibration error of the maximum-probability prediction."""
    confidences = probabilities.max(axis=-1)
    correct = (probabilities.argmax(axis=-1) == targets).astype(float)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for low, high in itertools.pairwise(edges):
        mask = (confidences > low) & (confidences <= high)
        if mask.any():
            ece += mask.mean() * abs(confidences[mask].mean() - correct[mask].mean())
    return float(ece)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    high = np.cumsum(counts)
    low = high - counts + 1
    return ((low + high) / 2.0)[inverse]


def auroc(scores_negative: np.ndarray, scores_positive: np.ndarray) -> float:
    """Rank-based AUROC for separating positives (higher scores) from negatives."""
    ranks = _average_ranks(np.concatenate([scores_negative, scores_positive]))
    n_neg, n_pos = len(scores_negative), len(scores_positive)
    u = ranks[n_neg:].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_neg * n_pos))
