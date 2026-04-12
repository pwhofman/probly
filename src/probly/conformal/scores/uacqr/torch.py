"""Torch implementation for UACQR scores."""

from __future__ import annotations

import torch

from probly.representation.sample.torch import TorchTensorSample

from ._common import uacqr_score_func, _weight_func


@uacqr_score_func.register(torch.Tensor)
def _(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
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


@_weight_func.register(torch.Tensor)
def _(y_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if y_pred.ndim != 3 or y_pred.shape[2] != 2:
        msg = f"intervals must have shape (n_estimations, n_samples, 2), got {y_pred.shape}"
        raise ValueError(msg)
    std = torch.std(y_pred, dim=0, unbiased=False)
    return std[:, 0], std[:, 1]


@uacqr_score_func.register(TorchTensorSample)
def _(y_pred: TorchTensorSample, y_true: torch.Tensor) -> torch.Tensor:
    """UACQR nonconformity scores for TorchTensorSample."""
    return uacqr_score_func(y_pred.tensor, y_true)


@_weight_func.register(TorchTensorSample)
def _(y_pred: TorchTensorSample) -> tuple[torch.Tensor, torch.Tensor]:
    """Weight functions for TorchTensorSample."""
    return _weight_func(y_pred.tensor)
