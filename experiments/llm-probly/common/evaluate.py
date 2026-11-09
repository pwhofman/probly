"""Evaluate uncertainty scores against correctness labels (AUROC, AURC, bootstrap)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from probly.evaluation.selective_prediction import selective_prediction
from sklearn.metrics import roc_auc_score


def _prepare(uncertainty: ArrayLike, correct: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return uncertainty as floats and ``wrong`` as booleans (True = the model was wrong)."""
    u = np.asarray(uncertainty, dtype=float)
    wrong = ~np.asarray(correct, dtype=bool)
    if u.shape != wrong.shape or u.ndim != 1:
        msg = f"uncertainty and correct must be 1-D and equally long, got {u.shape} and {wrong.shape}"
        raise ValueError(msg)
    return u, wrong


def auroc(uncertainty: ArrayLike, correct: ArrayLike) -> float:
    """AUROC of the uncertainty for detecting wrong answers (wrong = positive class)."""
    u, wrong = _prepare(uncertainty, correct)
    return float(roc_auc_score(wrong, u))


def aurc(
    uncertainty: ArrayLike,
    correct: ArrayLike,
    n_bins: int = 10,
    n_tie_seeds: int = 20,
    seed: int = 0,
) -> float:
    """Area under the risk-coverage curve, ties broken randomly (lower is better).

    probly's selective_prediction breaks ties by input position. We shuffle the
    questions before each call and average over ``n_tie_seeds`` shuffles instead.
    """
    u, wrong = _prepare(uncertainty, correct)
    losses = wrong.astype(float)
    n_bins = min(n_bins, len(u))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_tie_seeds):
        perm = rng.permutation(len(u))
        value, _bin_losses = selective_prediction(u[perm], losses[perm], n_bins=n_bins)
        values.append(value)
    return float(np.mean(values))


def evaluate(
    uncertainty: ArrayLike,
    correct: ArrayLike,
    n_bins: int = 10,
    n_tie_seeds: int = 20,
    seed: int = 0,
) -> dict:
    """Return AUROC and tie-broken AURC for one method."""
    return {
        "auroc": auroc(uncertainty, correct),
        "aurc": aurc(uncertainty, correct, n_bins, n_tie_seeds, seed),
    }


def bootstrap_auroc(
    uncertainty: ArrayLike,
    correct: ArrayLike,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap confidence interval of the AUROC by resampling the questions."""
    u, wrong = _prepare(uncertainty, correct)
    rng = np.random.default_rng(seed)
    n = len(u)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = wrong[idx]
        if sample.all() or not sample.any():
            continue  # only one class in this resample, AUROC is undefined
        values.append(roc_auc_score(sample, u[idx]))
    if not values:
        msg = "all bootstrap samples contained only one class"
        raise ValueError(msg)
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
