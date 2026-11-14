"""Evaluate uncertainty scores against correctness labels (AUROC, AURC, PRR, ECE, bootstrap)."""

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


def _rejection_area(u: np.ndarray, quality: np.ndarray) -> float:
    """Area under the prediction-rejection curve (higher = better).

    Sort the questions from least to most uncertain. For every k, take the k
    least uncertain questions and compute their mean quality (here: accuracy).
    The area is the average of these k = 1..n accuracies. Same formula as
    lm-polygraph's ``PredictionRejectionArea`` with ``max_rejection=1``.
    """
    order = np.argsort(u, kind="stable")  # least uncertain first
    kept_quality = np.cumsum(quality[order])  # total quality of the k least uncertain
    mean_quality = kept_quality / np.arange(1, len(u) + 1)
    return float(mean_quality.mean())


def rejection_curve(
    uncertainty: ArrayLike,
    correct: ArrayLike,
    n_tie_seeds: int = 20,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Accuracy of the kept questions when the most uncertain ones are rejected.

    Returns ``(rejected_fraction, accuracy)`` with one point per number of
    rejected questions (0, 1, ..., n-1). The mean of ``accuracy`` is the area
    that ``prr`` uses. Ties are broken randomly and averaged, as in ``aurc``.
    """
    u, wrong = _prepare(uncertainty, correct)
    quality = (~wrong).astype(float)
    n = len(u)
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(n_tie_seeds):
        perm = rng.permutation(n)
        order = np.argsort(u[perm], kind="stable")  # least uncertain first
        kept = np.cumsum(quality[perm][order]) / np.arange(1, n + 1)  # accuracy of the k least uncertain
        curves.append(kept[::-1])  # index i = i questions rejected
    rejected = np.arange(n) / n
    return rejected, np.mean(curves, axis=0)


def prr(
    uncertainty: ArrayLike,
    correct: ArrayLike,
    n_tie_seeds: int = 20,
    seed: int = 0,
) -> float:
    """Prediction rejection ratio (higher is better): 1 = oracle, 0 = random, < 0 = worse than random.

    PRR = (area_method - area_random) / (area_oracle - area_random), as in
    lm-polygraph's ``prr_normalized``. The random area is exactly the accuracy
    (a random order keeps the accuracy at every rejection rate), so we use it
    directly instead of averaging random shuffles. Ties are broken randomly,
    averaged over ``n_tie_seeds`` shuffles, like in ``aurc``.
    """
    u, wrong = _prepare(uncertainty, correct)
    quality = (~wrong).astype(float)
    area_random = float(quality.mean())
    area_oracle = _rejection_area(-quality, quality)  # correct answers first
    if np.isclose(area_oracle, area_random):
        return float("nan")  # all answers right or all wrong: PRR is undefined
    rng = np.random.default_rng(seed)
    areas = []
    for _ in range(n_tie_seeds):
        perm = rng.permutation(len(u))
        areas.append(_rejection_area(u[perm], quality[perm]))
    return (float(np.mean(areas)) - area_random) / (area_oracle - area_random)


def ece(uncertainty: ArrayLike, correct: ArrayLike, n_bins: int = 10) -> float:
    """Expected calibration error (lower is better, 0 = perfectly calibrated).

    Only meaningful for methods whose score is a probability of being correct
    (p(True), verbalized confidence), with uncertainty = 1 - confidence.
    The confidences are put into ``n_bins`` equal-width bins; in every bin we
    compare the mean confidence with the accuracy, and average the gaps,
    weighted by how many questions fall into the bin. Same binning as
    lm-polygraph's ``ECE`` (first bin includes 0, the others are (lo, hi]).
    """
    u, wrong = _prepare(uncertainty, correct)
    confidence = 1.0 - u
    if np.any((confidence < 0) | (confidence > 1)):
        msg = "ECE needs uncertainty = 1 - probability, so values in [0, 1]"
        raise ValueError(msg)
    accuracy = (~wrong).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            in_bin = (confidence >= lo) & (confidence <= hi)
        else:
            in_bin = (confidence > lo) & (confidence <= hi)
        if not in_bin.any():
            continue
        gap = abs(accuracy[in_bin].mean() - confidence[in_bin].mean())
        total += in_bin.mean() * gap  # in_bin.mean() = share of questions in this bin
    return float(total)


def evaluate(
    uncertainty: ArrayLike,
    correct: ArrayLike,
    n_bins: int = 10,
    n_tie_seeds: int = 20,
    seed: int = 0,
) -> dict:
    """Return AUROC, tie-broken AURC and PRR for one method."""
    return {
        "auroc": auroc(uncertainty, correct),
        "aurc": aurc(uncertainty, correct, n_bins, n_tie_seeds, seed),
        "prr": prr(uncertainty, correct, n_tie_seeds, seed),
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


def rejection_curve(uncertainty, correct, n_bins=10, n_tie_seeds=20, seed=0):
    """Error rate of the kept questions after rejecting the most uncertain ones.

    Returns ``(rejected, error)``: ``rejected[i]`` is the fraction of questions
    rejected (0 to 1) and ``error[i]`` the error rate among the remaining ones.
    Ties are broken randomly, the result is averaged over ``n_tie_seeds`` runs.
    """
    u, wrong = _prepare(uncertainty, correct)
    losses = wrong.astype(float)
    n_bins = min(n_bins, len(u))
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(n_tie_seeds):
        perm = rng.permutation(len(u))
        _value, bin_losses = selective_prediction(u[perm], losses[perm], n_bins=n_bins)
        curves.append(np.asarray(bin_losses, dtype=float))
    return np.linspace(0, 1, n_bins), np.mean(curves, axis=0)
