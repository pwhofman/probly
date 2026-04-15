"""Torch implementation for CQR scores."""

from __future__ import annotations

import torch

from probly.representation.sample.torch import TorchSample

from ._common import cqr_score_func


@cqr_score_func.register(torch.Tensor)
def _(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """CQR nonconformity scores for PyTorch tensors."""
    y = y_true.reshape(-1)

    if y_pred.ndim != 2 or y_pred.shape[1] != 2:
        msg = f"y_pred must have shape (n_samples, 2), got {y_pred.shape}"
        raise ValueError(msg)

    lower = y_pred[:, 0]
    upper = y_pred[:, 1]

    return torch.maximum(lower - y, y - upper)


@cqr_score_func.register(TorchSample)
def _(y_pred: TorchSample, y_true: torch.Tensor) -> torch.Tensor:
    """CQR nonconformity scores for TorchSamples."""
    return cqr_score_func(y_pred.tensor, y_true)
