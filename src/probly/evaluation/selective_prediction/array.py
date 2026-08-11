"""NumPy implementation of the selective prediction evaluation task."""

from __future__ import annotations

import numpy as np

from ._common import selective_prediction


@selective_prediction.register(np.ndarray)
def selective_prediction_numpy(criterion: np.ndarray, losses: np.ndarray, n_bins: int = 50) -> tuple[float, np.ndarray]:
    """Perform selective prediction for NumPy arrays."""
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
    aurc = float(np.trapezoid(bin_losses, np.linspace(0, 1, n_bins)))
    return float(aurc), bin_losses
