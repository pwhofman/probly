"""Semantic calibration metrics and post-hoc calibrators.

Provides B-calibration measurement (ECE/ACE at the semantic level) and three
post-hoc calibration methods (temperature scaling, Platt scaling, isotonic
regression).  All calibrators share a ``fit`` / ``calibrate`` interface.

Reference: arXiv:2511.04869 -- "Semantic Calibration"
"""

from __future__ import annotations

from collections import Counter
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.isotonic import IsotonicRegression

from gemma_experiment.semantic_entropy import _logsumexp

if TYPE_CHECKING:
    from numpy.typing import NDArray

_EPS = 1e-7


# ---------------------------------------------------------------------------
# Semantic confidence
# ---------------------------------------------------------------------------


def compute_semantic_confidence_discrete(
    semantic_ids: list[int],
) -> tuple[int, float]:
    """Mode cluster and its count-based probability.

    Returns:
        ``(mode_cluster_id, confidence)`` where
        ``confidence = count(mode) / N``.
    """
    counts = Counter(semantic_ids)
    mode_id, mode_count = counts.most_common(1)[0]
    return mode_id, mode_count / len(semantic_ids)


def compute_semantic_confidence_weighted(
    semantic_ids: list[int],
    log_likelihoods: list[float],
) -> tuple[int, float]:
    """Mode cluster and its log-likelihood-weighted probability.

    Cluster probabilities are computed via logsumexp, matching
    ``weighted_semantic_entropy``.

    Returns:
        ``(mode_cluster_id, confidence)`` where confidence is the
        normalized probability of the highest-probability cluster.
    """
    ll = np.array(log_likelihoods)
    log_normalizer = _logsumexp(ll)

    unique_ids = sorted(set(semantic_ids))
    best_id, best_log_p = unique_ids[0], -np.inf
    for uid in unique_ids:
        member_lls = ll[[i for i, sid in enumerate(semantic_ids) if sid == uid]]
        log_p = _logsumexp(member_lls) - log_normalizer
        if log_p > best_log_p:
            best_log_p = log_p
            best_id = uid

    return best_id, float(np.exp(best_log_p))


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def expected_calibration_error(
    confidences: NDArray[np.floating],
    correctness: NDArray[np.floating],
    n_bins: int = 10,
) -> float:
    """Binned Expected Calibration Error.

    Weighted average of ``|accuracy - confidence|`` per bin.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for k, (lo, hi) in enumerate(pairwise(bin_edges)):
        mask = (confidences >= lo) & (confidences <= hi) if k == 0 else (confidences > lo) & (confidences <= hi)
        count = mask.sum()
        if count == 0:
            continue
        acc = correctness[mask].mean()
        conf = confidences[mask].mean()
        ece += (count / n) * abs(acc - conf)
    return float(ece)


def average_calibration_error(
    confidences: NDArray[np.floating],
    correctness: NDArray[np.floating],
) -> float:
    """Unbinned Average Calibration Error: ``mean(|confidence - correctness|)``.

    Better suited for small sample sizes where binning is unreliable.
    """
    return float(np.mean(np.abs(confidences - correctness)))


def reliability_diagram_data(
    confidences: NDArray[np.floating],
    correctness: NDArray[np.floating],
    n_bins: int = 10,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.integer]]:
    """Return ``(bin_centers, bin_accuracies, bin_counts)`` for plotting."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    accuracies = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for k, (lo, hi) in enumerate(pairwise(bin_edges)):
        mask = (confidences >= lo) & (confidences <= hi) if k == 0 else (confidences > lo) & (confidences <= hi)
        counts[k] = mask.sum()
        if counts[k] > 0:
            accuracies[k] = correctness[mask].mean()
    return centers, accuracies, counts


# ---------------------------------------------------------------------------
# Post-hoc calibrators
# ---------------------------------------------------------------------------


def _logit(p: NDArray[np.floating]) -> NDArray[np.floating]:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(x: NDArray[np.floating]) -> NDArray[np.floating]:
    return 1.0 / (1.0 + np.exp(-x))


class TemperatureScaler:
    """Single-parameter temperature scaling on logit space."""

    def __init__(self) -> None:
        """Initialize with identity temperature (T=1)."""
        self.temperature: float = 1.0

    def fit(
        self,
        confidences: NDArray[np.floating],
        correctness: NDArray[np.floating],
    ) -> None:
        """Optimize T to minimize negative log-likelihood."""
        logits = _logit(confidences)

        def nll(t: float) -> float:
            p = np.clip(_sigmoid(logits / t), _EPS, 1 - _EPS)
            return float(-np.mean(correctness * np.log(p) + (1 - correctness) * np.log(1 - p)))

        result = minimize_scalar(nll, bounds=(0.01, 20.0), method="bounded")
        self.temperature = float(result.x)

    def calibrate(self, confidences: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply temperature scaling to confidence scores."""
        return _sigmoid(_logit(confidences) / self.temperature)


class PlattScaler:
    """Two-parameter Platt scaling: ``sigmoid(a * logit(conf) + b)``."""

    def __init__(self) -> None:
        """Initialize with identity parameters (a=1, b=0)."""
        self.a: float = 1.0
        self.b: float = 0.0

    def fit(
        self,
        confidences: NDArray[np.floating],
        correctness: NDArray[np.floating],
    ) -> None:
        """Optimize (a, b) to minimize negative log-likelihood."""
        logits = _logit(confidences)

        def nll(params: NDArray[np.floating]) -> float:
            a, b = params
            p = np.clip(_sigmoid(a * logits + b), _EPS, 1 - _EPS)
            return float(-np.mean(correctness * np.log(p) + (1 - correctness) * np.log(1 - p)))

        result = minimize(nll, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
        self.a, self.b = float(result.x[0]), float(result.x[1])

    def calibrate(self, confidences: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply Platt scaling to confidence scores."""
        return _sigmoid(self.a * _logit(confidences) + self.b)


class IsotonicCalibrator:
    """Non-parametric isotonic regression calibrator."""

    def __init__(self) -> None:
        """Initialize isotonic regression with [0, 1] bounds."""
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(
        self,
        confidences: NDArray[np.floating],
        correctness: NDArray[np.floating],
    ) -> None:
        """Fit isotonic regression on confidence-correctness pairs."""
        self._iso.fit(confidences, correctness)

    def calibrate(self, confidences: NDArray[np.floating]) -> NDArray[np.floating]:
        """Apply isotonic regression to confidence scores."""
        return self._iso.predict(confidences)


CALIBRATORS: list[tuple[str, type]] = [
    ("Temperature", TemperatureScaler),
    ("Platt", PlattScaler),
    ("Isotonic", IsotonicCalibrator),
]

# ---------------------------------------------------------------------------
# LOOCV evaluation
# ---------------------------------------------------------------------------


def leave_one_out_evaluate(
    confidences: NDArray[np.floating],
    correctness: NDArray[np.floating],
    calibrator_cls: type,
) -> tuple[float, NDArray[np.floating]]:
    """Leave-one-out cross-validated calibration evaluation.

    For each sample, trains on N-1, predicts the held-out one.

    Returns:
        ``(loocv_ace, calibrated_confidences)``
    """
    n = len(confidences)
    calibrated = np.zeros(n)
    for i in range(n):
        train_conf = np.concatenate([confidences[:i], confidences[i + 1 :]])
        train_corr = np.concatenate([correctness[:i], correctness[i + 1 :]])
        cal = calibrator_cls()
        cal.fit(train_conf, train_corr)
        calibrated[i] = cal.calibrate(confidences[i : i + 1])[0]

    ace = average_calibration_error(calibrated, correctness)
    return ace, calibrated


def compute_aggregates(
    results: list[dict],
    correctness_key_discrete: str = "is_correct_discrete",
    correctness_key_weighted: str = "is_correct_weighted",
) -> dict:
    """Compute aggregate calibration metrics from per-question results.

    Args:
        results: List of per-question result dicts. Each must contain
            ``confidence_discrete``, ``confidence_weighted``, and the
            correctness keys.
        correctness_key_discrete: Key for discrete correctness values.
        correctness_key_weighted: Key for weighted correctness values.
    """
    conf_d = np.array([r["confidence_discrete"] for r in results])
    conf_w = np.array([r["confidence_weighted"] for r in results])
    corr_d = np.array([float(r[correctness_key_discrete]) for r in results])
    corr_w = np.array([float(r[correctness_key_weighted]) for r in results])

    return {
        "ece_discrete": float(expected_calibration_error(conf_d, corr_d)),
        "ece_weighted": float(expected_calibration_error(conf_w, corr_w)),
        "ace_discrete": float(average_calibration_error(conf_d, corr_d)),
        "ace_weighted": float(average_calibration_error(conf_w, corr_w)),
        "accuracy": float(corr_d.mean()),
    }
