"""Torch implementation for UACQR scores."""

from __future__ import annotations

import torch

from probly.representation.sample.torch import TorchSample

from ._common import uacqr_score_func


@uacqr_score_func.register(torch.Tensor)
def compute_uacqr_score_func_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """UACQR nonconformity scores for PyTorch tensors."""
    y = y_true.reshape(-1)

    if y_pred.ndim != 3 or y_pred.shape[2] != 2:
        msg = f"intervals must have shape (n_estimations, n_samples, 2), got {y_pred.shape}"
        raise ValueError(msg)

    std = torch.std(y_pred, dim=0, unbiased=False)
    mean_intervals = torch.mean(y_pred, dim=0)

    lower = mean_intervals[:, 0]
    upper = mean_intervals[:, 1]
    std_lo = std[:, 0]
    std_hi = std[:, 1]

    return torch.maximum((lower - y) / std_lo, (y - upper) / std_hi)


@uacqr_score_func.register(TorchSample)
def _(y_pred: TorchSample, y_true: torch.Tensor) -> torch.Tensor:
    """UACQR nonconformity scores for TorchSample."""
    return compute_uacqr_score_func_torch(y_pred.tensor, y_true)
