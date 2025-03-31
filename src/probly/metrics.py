import numpy as np


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """Compute the expected calibration error (ECE) of the predicted probabilities.

    Args:
        probs: numpy.ndarray of shape (n_instances, n_classes)
        labels: numpy.ndarray of shape (n_instances,)
        num_bins: int
    Returns:
        ece: float

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

    return ece
