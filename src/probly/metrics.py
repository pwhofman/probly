"""Collection of performance metrics to evaluate predictions."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

__all__ = [
    "coverage",
    "efficiency",
    "expected_calibration_error",
]


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """Compute the expected calibration error (ECE) of the predicted probabilities.

    Args:
        probs: The predicted probabilities as an array of shape (n_instances, n_classes).
        labels: The true labels as an array of shape (n_instances,).
        num_bins: The number of bins to use for the calibration error calculation.

    Returns:
        ece: The expected calibration error.
    """
    confs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    bins = np.linspace(0, 1, num_bins + 1, endpoint=True)
    bin_indices = np.digitize(confs, bins, right=True) - 1
    num_instances = probs.shape[0]
    ece = 0
    for i in range(num_bins):
        _bin = np.where(bin_indices == i)[0]
        # check if bin is empty
        if _bin.shape[0] == 0:
            continue
        acc_bin = np.mean(preds[_bin] == labels[_bin])
        conf_bin = np.mean(confs[_bin])
        weight = _bin.shape[0] / num_instances
        ece += weight * np.abs(acc_bin - conf_bin)
    return float(ece)


def coverage(preds: np.ndarray, targets: np.ndarray) -> float:
    """Compute the coverage of set-valued predictions.

    Args:
        preds: The predictions as an array of shape `(n_instances, n_classes)` or
            `(n_instances, n_samples, n_classes)`
        targets: The targets as an array of shape `(n_instances,)` or `(n_instances, n_classes)`

    Returns:
        cov: The coverage of the set-valued predictions

    """
    if preds.ndim == 2:
        cov = np.mean(preds[np.arange(preds.shape[0]), targets])
    elif preds.ndim == 3:
        probs_lower = np.round(np.nanmin(preds, axis=1), decimals=3)
        probs_upper = np.round(np.nanmax(preds, axis=1), decimals=3)
        covered = np.all((probs_lower <= targets) & (targets <= probs_upper), axis=1)
        cov = np.mean(covered)
    else:
        raise ValueError(f"Expected 2D or 3D array, got {preds.ndim}D")
    return float(cov)


def efficiency(preds: np.ndarray) -> float:
    """Compute the efficiency of set-valued predictions.

    In the case of a set over classes this is the mean of the number of classes in the set. In the
    case of a credal set, this is computed by the mean difference between the upper and lower
    probabilities.

    Args:
        preds: The predictions as an array of shape `(n_instances, n_classes)` or
            of shape `(n_instances, n_samples, n_classes)`.

    Returns:
        eff: The efficiency of the set-valued predictions.

    """
    if preds.ndim == 2:
        eff = np.mean(preds)
    elif preds.ndim == 3:
        probs_lower = np.round(np.nanmin(preds, axis=1), decimals=3)
        probs_upper = np.round(np.nanmax(preds, axis=1), decimals=3)
        eff = np.mean(probs_upper - probs_lower)
    else:
        raise ValueError(f"Expected 2D or 3D array, got {preds.ndim}D")
    return float(eff)


def coverage_convex_hull(probs: np.ndarray, targets: np.ndarray) -> float:
    """Compute the coverage given the credal set defined by the convex hull of the predicted probabilities.

    The coverage is defined as the proportion of instances whose true distribution is contained in the convex hull.
    This is computed using linear programming by checking whether the target distribution can be expressed as
    a convex combination of the predicted distributions.

    Args:
        probs: The predicted probabilities as an array of shape (n_instances, n_samples, n_classes).
        targets: The true labels as an array of shape (n_instances, n_classes).

    Returns:
        cov: The coverage.
    """
    covered = 0
    n_extrema = probs.shape[1]
    c = np.zeros(n_extrema)  # we do not care about the coefficients in this case
    bounds = [(0, 1)] * n_extrema
    for i in range(probs.shape[0]):
        a_eq = np.vstack((probs[i].T, np.ones(n_extrema)))
        b_eq = np.concatenate((targets[i], [1]), axis=0)
        res = linprog(c=c, A_eq=a_eq, b_eq=b_eq, bounds=bounds)
        covered += res.success
    cov = covered / probs.shape[0]
    return float(cov)


def covered_efficiency(preds: np.ndarray, targets: np.ndarray) -> float:
    """Compute the efficiency of the set-valued predictions for which the ground truth is covered.

    In the case of a set over classes this is the mean of the number of classes in the set. In the
    case of a credal set, this is computed by the mean difference between the upper and lower
    probabilities.

    Args:
        preds: The predictions as an array of shape `(n_instances, n_classes)` or
            of shape `(n_instances, n_samples, n_classes)`.
        targets: The true labels as an array of shape (n_instances, n_classes).

    Returns:
        ceff: The efficiency of the set-valued predictions for which the ground truth is covered.

    """
    if preds.ndim == 2:
        covered = preds[np.arange(preds.shape[0]), targets]
        ceff = np.mean(preds[covered])
    elif preds.ndim == 3:
        probs_lower = np.round(np.nanmin(preds, axis=1), decimals=3)
        probs_upper = np.round(np.nanmax(preds, axis=1), decimals=3)
        covered = np.all((probs_lower <= targets) & (targets <= probs_upper), axis=1)
        ceff = np.mean((probs_upper - probs_lower)[covered])
    else:
        raise ValueError(f"Expected 2D or 3D array, got {preds.ndim}D")
    return float(ceff)


def log_loss(probs: np.ndarray, targets: np.ndarray) -> float | np.ndarray:
    """Compute the log loss of the predicted probabilities.

    Args:
        probs: The predicted probabilities as an array of shape (n_instances, n_classes).
        targets: The true labels as an array of shape (n_instances,). If None, the log loss is
            computed for all classes. This can be used for uncertainty quantification.

    Returns:
        loss: The log loss.
    """
    if targets is None:
        loss = -np.log(probs)
    else:
        loss = -np.log(probs[np.arange(probs.shape[0]), targets])
        loss = np.mean(loss)
    return loss


def brier_score(probs: np.ndarray, targets: np.ndarray) -> float | np.ndarray:
    """Compute the Brier score of the predicted probabilities.

    We assume the score to be negatively-oriented, i.e. lower is better.

    Args:
        probs: The predicted probabilities as an array of shape (n_instances, n_classes).
        targets: The true labels as an array of shape (n_instances,). If None, the Brier score is
            computed for all classes. This can be used for uncertainty quantification.

    Returns:
        loss: The Brier score.
    """
    if targets is None:
        loss = 1 - 2 * probs + np.sum(probs**2, -1)[..., None]
    else:
        loss = 1 - 2 * probs[np.arange(probs.shape[0]), targets] + np.sum(probs**2, axis=1)
        loss = np.mean(loss)
    return loss


def zero_one_loss(probs: np.ndarray, targets: np.ndarray) -> float | np.ndarray:
    """Compute the zero-one loss of the predicted probabilities.

    Args:
        probs: The predicted probabilities as an array of shape (n_instances, n_classes).
        targets: The true labels as an array of shape (n_instances,). If None, the zero-one loss is
            computed for all classes. This can be used for uncertainty quantification.

    Returns:
        loss: The zero-one loss.
    """
    if targets is None:
        loss = probs != np.max(probs, axis=-1)[..., None]
    else:
        loss = np.argmax(probs, axis=1) != targets
        loss = np.mean(loss)
    return loss


def spherical_score(probs: np.ndarray, targets: np.ndarray) -> float | np.ndarray:
    """Compute the spherical score of the predicted probabilities.

    We assume the score to be negatively-oriented, i.e. lower is better.

    Args:
        probs: The predicted probabilities as an array of shape (n_instances, n_classes).
        targets: The true labels as an array of shape (n_instances,). If None, the spherical score is
            computed for all classes. This can be used for uncertainty quantification.

    Returns:
        loss: The spherical score.
    """
    if targets is None:
        loss = 1 - probs / np.sqrt(np.sum(probs**2, -1))[..., None]
    else:
        loss = 1 - probs[np.arange(probs.shape[0]), targets] / np.sqrt(np.sum(probs**2, axis=1))
        loss = np.mean(loss)
    return loss
