"""Evaluation metrics for active learning experiments."""

from __future__ import annotations

import numpy as np
import torch

from probly.train.calibration.torch import ExpectedCalibrationError


def compute_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute classification accuracy.

    Args:
        y_pred: Predicted class labels, shape (n,).
        y_true: Ground-truth class labels, shape (n,).

    Returns:
        Fraction of correct predictions in [0, 1].
    """
    return float(np.mean(y_pred == y_true))


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error.

    Wraps probly.train.calibration.torch.ExpectedCalibrationError.
    Converts numpy inputs to torch tensors, runs ECE, and returns a float.

    Args:
        probs: Class probability matrix of shape (n, n_classes).
        y_true: Ground-truth integer class labels of shape (n,).
        n_bins: Number of confidence bins for the ECE histogram.

    Returns:
        ECE value in [0, 1].
    """
    inputs = torch.as_tensor(probs, dtype=torch.float32)
    targets = torch.as_tensor(y_true, dtype=torch.long)
    ece_fn = ExpectedCalibrationError(num_bins=n_bins)
    with torch.no_grad():
        loss = ece_fn(inputs, targets)
    return float(loss.item())


def compute_nauc(scores: list[float]) -> float:
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
