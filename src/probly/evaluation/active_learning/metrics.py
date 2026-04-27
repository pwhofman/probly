"""Evaluation metrics for active learning experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from flextype import flexdispatch

if TYPE_CHECKING:
    from collections.abc import Sequence


@flexdispatch
def compute_accuracy(y_pred: object, y_true: object) -> float:
    """Compute classification accuracy.

    Args:
        y_pred: Predicted class labels, shape (n,).
        y_true: Ground-truth class labels, shape (n,).

    Returns:
        Fraction of correct predictions in [0, 1].
    """
    msg = f"No compute_accuracy implementation registered for type {type(y_pred)}"
    raise NotImplementedError(msg)


@flexdispatch
def compute_ece(probs: object, y_true: object, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error.

    Args:
        probs: Class probability matrix of shape (n, n_classes).
        y_true: Ground-truth integer class labels of shape (n,).
        n_bins: Number of confidence bins for the ECE histogram.

    Returns:
        ECE value in [0, 1].
    """
    msg = f"No compute_ece implementation registered for type {type(probs)}"
    raise NotImplementedError(msg)


# ---------------------------------------------------------------------------
# NumPy registrations (always available)
# ---------------------------------------------------------------------------


@compute_accuracy.register(np.ndarray)
def _compute_accuracy_numpy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(y_pred == y_true))


@compute_ece.register(np.ndarray)
def _compute_ece_numpy(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.searchsorted(bins, confs, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    n_samples = len(probs)
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        count = mask.sum()
        if count == 0:
            continue
        acc = float(np.mean(preds[mask] == y_true[mask]))
        conf = float(np.mean(confs[mask]))
        ece += (count / n_samples) * abs(acc - conf)
    return ece


# ---------------------------------------------------------------------------
# PyTorch registrations (lazy)
# ---------------------------------------------------------------------------

try:
    import torch

    from probly.train.calibration.torch import ExpectedCalibrationError

    @compute_accuracy.register(torch.Tensor)
    def _compute_accuracy_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        return float((y_pred == y_true).float().mean().item())

    @compute_ece.register(torch.Tensor)
    def _compute_ece_torch(probs: torch.Tensor, y_true: torch.Tensor, n_bins: int = 10) -> float:
        ece_fn = ExpectedCalibrationError(num_bins=n_bins)
        with torch.no_grad():
            loss = ece_fn(probs.float(), y_true.long())
        return float(loss.item())

except ImportError:
    pass


# ---------------------------------------------------------------------------
# Backend-agnostic metrics
# ---------------------------------------------------------------------------


def compute_nauc(scores: Sequence[float]) -> float:
    """Compute normalized area under the score curve.

    Normalizes by the ideal AUC, which is the area under a constant curve at
    max(scores). NaN entries (e.g. from an exhausted pool) are excluded while
    preserving their original iteration indices so that x-axis spacing remains
    correct.

    Args:
        scores: Per-iteration scores in [0, 1], higher-is-better.

    Returns:
        Normalized AUC in [0, 1], or NaN if fewer than two finite scores are
        available. Returns 1.0 when the ideal AUC is zero and all finite
        scores are equal.
    """
    s = np.asarray(scores, dtype=float)
    x = np.arange(len(s), dtype=float)
    mask = np.isfinite(s)
    if mask.sum() < 2:
        return float("nan")
    s_valid = s[mask]
    x_valid = x[mask]
    actual_auc = float(np.trapezoid(s_valid, x=x_valid))
    ideal_auc = float(x_valid[-1] - x_valid[0])
    if ideal_auc == 0.0:
        # Only one unique x position: all values equal, return 1.0.
        return 1.0
    return actual_auc / ideal_auc
