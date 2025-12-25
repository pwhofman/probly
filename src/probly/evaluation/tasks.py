"""Collection of downstream tasks to evaluate the performance of uncertainty pipelines."""

from __future__ import annotations

import numpy as np

# TODO(mmshlk): remove sklearn dependency - https://github.com/pwhofman/probly/issues/132
import sklearn.metrics as sm


def selective_prediction(criterion: np.ndarray, losses: np.ndarray, n_bins: int = 50) -> tuple[float, np.ndarray]:
    """Selective prediction downstream task for evaluation.

    Perform selective prediction based on criterion and losses.
    The criterion is used the sort the losses. In line with uncertainty
    literature the sorting is done in descending order, i.e.
    the losses with the largest criterion are rejected first.

    Args:
        criterion: numpy.ndarray shape (n_instances,), criterion values
        losses: numpy.ndarray shape (n_instances,), loss values
        n_bins: int, number of bins
    Returns:
        auroc: float, area under the loss curve
        bin_losses: numpy.ndarray shape (n_bins,), loss per bin

    """
    if n_bins > len(losses):
        msg = "The number of bins can not be larger than the number of elements criterion"
        raise ValueError(msg)
    sort_idxs = np.argsort(criterion)[::-1]
    losses_sorted = losses[sort_idxs]
    bin_len = len(losses) // n_bins
    bin_losses = np.empty(n_bins)
    for i in range(n_bins):
        bin_losses[i] = np.mean(losses_sorted[(i * bin_len) :])

    # Also compute the area under the loss curve based on the bin losses.
    auroc = sm.auc(np.linspace(0, 1, n_bins), bin_losses)
    return float(auroc), bin_losses


def out_of_distribution_detection_auroc(in_distribution: np.ndarray, out_distribution: np.ndarray) -> float:
    """Perform out-of-distribution detection using prediction functionals from id and ood data.

    This can be epistemic uncertainty, as is common, but also e.g. softmax confidence.

    Args:
        in_distribution: in-distribution prediction functionals
        out_distribution: out-of-distribution prediction functionals
    Returns:
        auroc: float, area under the roc curve

    """
    preds = np.concatenate((in_distribution, out_distribution))
    labels = np.concatenate((np.zeros(len(in_distribution)), np.ones(len(out_distribution))))
    auroc = sm.roc_auc_score(labels, preds)
    return float(auroc)


def out_of_distribution_detection_aupr(in_distribution: np.ndarray, out_distribution: np.ndarray) -> float:
    """Perform out-of-distribution detection using AUPR (Area Under the Precision-Recall Curve).

    This metric evaluates how well the model distinguishes between in- and out-of-distribution samples,
    focusing more on positive class (OOD) precision and recall.

    Args:
        in_distribution: in-distribution prediction functionals
        out_distribution: out-of-distribution prediction functionals

    Returns:
        aupr: float, area under the precision-recall curve
    """
    preds = np.concatenate((in_distribution, out_distribution))
    labels = np.concatenate((np.zeros(len(in_distribution)), np.ones(len(out_distribution))))
    aupr = sm.average_precision_score(labels, preds)
    return float(aupr)


def out_of_distribution_detection_fpr_at_x_tpr(
    in_distribution: np.ndarray,
    out_distribution: np.ndarray,
    tpr_target: float = 0.95,
) -> float:
    """Perform out-of-distribution detection using false positive rate (FPR) at a given true positive rate.

    If no thresholds are specified, the default tpr_target is 0.95.

    This can be epistemic uncertainty, as is common, but also e.g. softmax confidence.

    Args:
        in_distribution: numpy.ndarray, scores for in-distribution samples
        out_distribution: numpy.ndarray, scores for out-of-distribution samples
        tpr_target: target TPR value in [0, 1], e.g. 0.95

    Returns:
        fpr_at_target: float, FPR at the first threshold where TPR >= tpr_target

    Notes:
        - Assumes that larger scores correspond to the positive class
          (out-of-distribution).
        - If tpr_target cannot be reached, a ValueError is raised.
    """
    if not 0.0 < tpr_target <= 1.0:
        msg = f"tpr_target must be in the interval (0, 1], got {tpr_target}."
        raise ValueError(msg)

    preds = np.concatenate((in_distribution, out_distribution))
    labels = np.concatenate(
        (np.zeros(len(in_distribution)), np.ones(len(out_distribution))),
    )

    fpr, tpr, _ = sm.roc_curve(labels, preds)

    idxs = np.where(tpr >= tpr_target)[0]
    if len(idxs) == 0:
        msg = f"Could not achieve TPR >= {tpr_target:.3f} with given scores."
        raise ValueError(msg)

    first_idx = idxs[0]
    fpr_at_target = fpr[first_idx]
    return float(fpr_at_target)


def out_of_distribution_detection_fnr_at_x_tpr(
    in_distribution: np.ndarray,
    out_distribution: np.ndarray,
    tpr_target: float = 0.95,
) -> float:
    """Perform out-of-distribution detection using false negative rate at user given true positive rate.

    If no thresholds are specified, the default tpr_target is 0.95.

    Args:
        in_distribution: in-distribution prediction functionals
        out_distribution: out-of-distribution prediction functionals
        tpr_target: target TPR value in [0, 1], e.g. 0.95

    Returns:
        fnr@X: float, FNR at the first threshold where TPR >= tpr_target
    """
    preds = np.concatenate((in_distribution, out_distribution))
    labels = np.concatenate((np.zeros(len(in_distribution)), np.ones(len(out_distribution))))
    _, tpr, _ = sm.roc_curve(labels, preds)
    idx = np.where(tpr >= tpr_target)[0]
    return float(1.0 - tpr[idx[0]]) if len(idx) else 1.0
