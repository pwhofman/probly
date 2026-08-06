"""PyTorch implementation of the selective prediction evaluation task."""

from __future__ import annotations

import torch

from ._common import selective_prediction


@selective_prediction.register(torch.Tensor)
def selective_prediction_torch(
    criterion: torch.Tensor, losses: torch.Tensor, n_bins: int = 50
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform selective prediction for PyTorch tensors."""
    if n_bins > losses.shape[0]:
        msg = "The number of bins can not be larger than the number of elements criterion"
        raise ValueError(msg)
    sort_idxs = torch.argsort(criterion, descending=True)
    losses_sorted = losses[sort_idxs]
    bin_len = losses.shape[0] // n_bins
    bin_losses = torch.stack([losses_sorted[(i * bin_len) :].mean() for i in range(n_bins)])
    aurc = torch.trapezoid(bin_losses, torch.linspace(0, 1, n_bins, dtype=bin_losses.dtype, device=bin_losses.device))
    return aurc, bin_losses
