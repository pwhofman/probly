import numpy as np
import sklearn.metrics as sm

def selective_prediction(criterion, losses, num_bins=50):
    """
    Perform selective prediction based on criterion and losses.
    The criterion is used the sort the losses. In line with uncertainty
    literature the sorting is done in descending order, i.e.
    the losses with the largest criterion are rejected first.
    """
    if num_bins > len(losses):
        raise ValueError('The number of bins can not be larger than the number of elements criterion')
    sort_idxs = np.argsort(criterion)[::-1]
    losses_sorted = losses[sort_idxs]
    bin_len = len(losses) // num_bins
    bin_losses = np.empty(num_bins)
    for i in range(num_bins):
        bin_losses[i] = np.mean(losses_sorted[(i * bin_len):])

    # Also compute the area under the loss curve based on the bin losses.
    auroc = sm.auc(np.linspace(0, 1, num_bins), bin_losses)
    return auroc, bin_losses


def out_of_distribution_detection(id, ood):
    """
    Perform out-of-distribution detection using prediction functionals from id and ood data.
    This can be epistemic uncertainty, as is common, but also e.g. softmax confidence.
    """
    preds = np.concatenate((id, ood))
    labels = np.concatenate((np.zeros(len(id)), np.ones(len(ood))))
    auroc = sm.roc_auc_score(labels, preds)
    return auroc
