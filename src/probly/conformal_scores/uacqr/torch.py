"""Torch implementation for UACQR scores."""

from __future__ import annotations

import torch

from probly.representation.sample.torch import TorchSample

from ._common import uacqr_score


@uacqr_score.register(torch.Tensor)
def compute_uacqr_score_func_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """UACQR nonconformity scores for PyTorch tensors."""
    y = y_true.to(dtype=torch.float)
    pred = y_pred.to(dtype=torch.float)

    if pred.ndim != y.ndim + 2 or pred.shape[-1] != 2:
        msg = (
            "intervals must have shape (n_estimations, ..., 2) with batch shape matching y_true; "
            f"got y_pred shape {tuple(pred.shape)} and y_true shape {tuple(y.shape)}."
        )
        raise ValueError(msg)

    std = torch.std(pred, dim=0, unbiased=False)
    mean_intervals = torch.mean(pred, dim=0)

    lower = mean_intervals[..., 0]
    upper = mean_intervals[..., 1]
    std_lo = std[..., 0]
    std_hi = std[..., 1]

    return torch.maximum((lower - y) / std_lo, (y - upper) / std_hi)


@uacqr_score.register(TorchSample)
def _(y_pred: TorchSample, y_true: torch.Tensor) -> torch.Tensor:
    """UACQR nonconformity scores for TorchSample."""
    return compute_uacqr_score_func_torch(y_pred.tensor, y_true)
