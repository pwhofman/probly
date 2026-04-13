"""Collection of performance metrics to evaluate predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linprog
from tqdm import tqdm

ROUND_DECIMALS = 3  # Number of decimals to round probabilities to when computing coverage, efficiency, etc.


def coverage(preds: np.ndarray, targets: np.ndarray) -> float:
    """Compute the coverage of set-valued predictions described in :cite:`angelopoulosGentleIntroduction2021`.

    Args:
        preds: Predictions of shape (n_instances, n_classes) or (n_instances, n_samples, n_classes).
        targets: Targets of shape (n_instances,) or (n_instances, n_classes).

    Returns:
        cov: The coverage of the set-valued predictions.

    """
    if preds.ndim == 2:
        cov = np.mean(preds[np.arange(preds.shape[0]), targets])
    elif preds.ndim == 3:
        probs_lower = np.round(np.nanmin(preds, axis=1), decimals=ROUND_DECIMALS)
        probs_upper = np.round(np.nanmax(preds, axis=1), decimals=ROUND_DECIMALS)
        covered = np.all((probs_lower <= targets) & (targets <= probs_upper), axis=1)
        cov = np.mean(covered)
    else:
        msg = f"Expected 2D or 3D array, got {preds.ndim}D"
        raise ValueError(msg)
    return float(cov)


def efficiency(preds: np.ndarray) -> float:
    """Compute the efficiency of set-valued predictions described in :cite:`angelopoulosGentleIntroduction2021`.

    In the case of a set over classes this is the mean of the number of classes in the set. In the
    case of a credal set, this is computed by the mean difference between the upper and lower
    probabilities.

    Args:
        preds: Predictions of shape (n_instances, n_classes) or (n_instances, n_samples, n_classes).

    Returns:
        eff: The efficiency of the set-valued predictions.

    """
    if preds.ndim == 2:
        eff = 1 - np.mean(preds)
    elif preds.ndim == 3:
        probs_lower = np.round(np.nanmin(preds, axis=1), decimals=ROUND_DECIMALS)
        probs_upper = np.round(np.nanmax(preds, axis=1), decimals=ROUND_DECIMALS)
        eff = 1 - np.mean(probs_upper - probs_lower)
    else:
        msg = f"Expected 2D or 3D array, got {preds.ndim}D"
        raise ValueError(msg)
    return float(eff)


def coverage_convex_hull(probs: np.ndarray, targets: np.ndarray, **kwargs: Any) -> float:  # noqa: ANN401
    """Compute credal set coverage via convex hull :cite:`nguyenCredalEnsembling2025`.

    The coverage is defined as the proportion of instances whose true distribution is contained in the convex hull.
    This is computed using linear programming by checking whether the target distribution can be expressed as
    a convex combination of the predicted distributions.

    Args:
        probs: The predicted probabilities as an array of shape (n_instances, n_samples, n_classes).
        targets: The true labels as an array of shape (n_instances, n_classes).
        **kwargs: Additional keyword arguments for the linear programming solver, e.g. tolerance.

    Returns:
        cov: The coverage.

    """
    covered = 0
    n_extrema = probs.shape[1]
    c = np.zeros(n_extrema)  # we do not care about the coefficients in this case
    bounds = [(0, 1)] * n_extrema
    for i in tqdm(range(probs.shape[0]), desc="Instances"):
        a_eq = np.vstack((probs[i].T, np.ones(n_extrema)))
        b_eq = np.concatenate((targets[i], [1]), axis=0)
        res = linprog(c=c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, **kwargs)
        covered += res.success
    cov = covered / probs.shape[0]
    return float(cov)


def covered_efficiency(preds: np.ndarray, targets: np.ndarray) -> float:
    """Compute the efficiency of the set-valued predictions for which the ground truth is covered.

    In the case of a set over classes this is the mean of the number of classes in the set. In the
    case of a credal set, this is computed by the mean difference between the upper and lower
    probabilities.

    Args:
        preds: Predictions of shape (n_instances, n_classes) or (n_instances, n_samples, n_classes).
        targets: The true labels as an array of shape (n_instances, n_classes).

    Returns:
        ceff: The efficiency of the set-valued predictions for which the ground truth is covered.

    """
    if preds.ndim == 2:
        covered = preds[np.arange(preds.shape[0]), targets]
        ceff = 1 - np.mean(preds[covered])
    elif preds.ndim == 3:
        probs_lower = np.round(np.nanmin(preds, axis=1), decimals=ROUND_DECIMALS)
        probs_upper = np.round(np.nanmax(preds, axis=1), decimals=ROUND_DECIMALS)
        covered = np.all((probs_lower <= targets) & (targets <= probs_upper), axis=1)
        ceff = 1 - np.mean((probs_upper - probs_lower)[covered])
    else:
        msg = f"Expected 2D or 3D array, got {preds.ndim}D"
        raise ValueError(msg)
    return float(ceff)
