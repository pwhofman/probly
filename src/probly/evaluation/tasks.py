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
    return auroc, bin_losses


def out_of_distribution_detection(in_distribution: np.ndarray, out_distribution: np.ndarray) -> float:
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

def out_of_distribution_detection_fpr_at_95_tpr(in_distribution: np.ndarray, out_distribution: np.ndarray) -> float:
    """Perform out-of-distribution detection using prediction functionals from id and ood data.

    This can be epistemic uncertainty, as is common, but also e.g. softmax confidence.

    Args:
        in_distribution: in-distribution prediction functionals
        out_distribution: out-of-distribution prediction functionals
    Returns:
        fpr@95tpr: float, false positive rate at 95% true positive rate
    """
    preds = np.concatenate((in_distribution, out_distribution))
    labels = np.concatenate((np.zeros(len(in_distribution)), np.ones(len(outdistribution))))
    fpr, tpr,  = sm.roc_curve(labels, preds, pos_label=0)

    target_tpr = 0.95

    if target_tpr in tpr:
        idx = np.where(tpr == target_tpr)[0][0]
        return float(fpr[idx])

    if target_tpr < tpr[0]:
        "tpr > 0.95"
        return float(fpr[0])
    "Interpolate fpr because tpr=0.95 usually does not occur exactly"
    return float(np.interp(target_tpr, tpr, fpr))

def out_of_distribution_detection_fnr_at_95(in_distribution: np.ndarray, out_distribution: np.ndarray) -> float:
    """Perform out-of-distribution detection using prediction functionals from id and ood data.

    This can be epistemic uncertainty, as is common, but also e.g. softmax confidence.

    Args:
        in_distribution: in-distribution prediction functionals
        out_distribution: out-of-distribution prediction functionals
    Returns:
        fnr@95: float, false negative rate at 95% true positive rate
    """
    preds = np.concatenate((in_distribution, out_distribution))
    labels = np.concatenate((np.zeros(len(in_distribution)), np.ones(len(out_distribution)))) 
    _, tpr, threshold = sm.roc_curve(labels, preds)

    target_tpr = 0.95
    idx = np.where(tpr >= target_tpr)[0][0]
    threshold_at_95_tpr = threshold[idx]


    labels_at_95 = (preds >= threshold_at_95_tpr).astype(int)
    fn = np.sum((labels == 1) & (labels_at_95 == 0))
    preds_pos = np.sum(labels == 1)
    fnr_at_95 = fn / preds_pos

    return float(fnr_at_95)
