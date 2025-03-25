import numpy as np
import sklearn.metrics as sm


def selective_prediction(criterion: np.ndarray, losses: np.ndarray, n_bins: int = 50) -> tuple[
    float, np.ndarray]:
    """
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
        raise ValueError('The number of bins can not be larger than the number of elements criterion')
    sort_idxs = np.argsort(criterion)[::-1]
    losses_sorted = losses[sort_idxs]
    bin_len = len(losses) // n_bins
    bin_losses = np.empty(n_bins)
    for i in range(n_bins):
        bin_losses[i] = np.mean(losses_sorted[(i * bin_len):])

    # Also compute the area under the loss curve based on the bin losses.
    auroc = sm.auc(np.linspace(0, 1, n_bins), bin_losses)
    return auroc, bin_losses


def out_of_distribution_detection(id: np.ndarray, ood: np.ndarray) -> float:
    """
    Perform out-of-distribution detection using prediction functionals from id and ood data.
    This can be epistemic uncertainty, as is common, but also e.g. softmax confidence.
    Args:
        id: numpy.ndarray shape (n_id_instances,), in-distribution prediction functionals
        ood: numpy.ndarray shape (n_ood_instances,), out-of-distribution prediction functionals
    Returns:
        auroc: float, area under the roc curve
    """
    preds = np.concatenate((id, ood))
    labels = np.concatenate((np.zeros(len(id)), np.ones(len(ood))))
    auroc = sm.roc_auc_score(labels, preds)
    return auroc
