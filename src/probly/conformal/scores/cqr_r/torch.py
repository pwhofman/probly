"""Torch implementation for CQR-r scores."""

from __future__ import annotations

import torch

from probly.representation.sample.torch import TorchTensorSample

from ._common import _EPS, cqr_r_score_func, weight_func


@cqr_r_score_func.register(torch.Tensor)
def _(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """CQR-r nonconformity scores for PyTorch tensors."""
    y = y_true.reshape(-1)

    if y_pred.ndim != 2 or y_pred.shape[1] != 2:
        msg = f"y_pred must have shape (n_samples, 2), got {y_pred.shape}"
        raise ValueError(msg)

    lower = y_pred[:, 0]
    upper = y_pred[:, 1]
    width = torch.clamp(upper - lower, min=_EPS)

    return torch.maximum(lower - y, y - upper) / width


@cqr_r_score_func.register(TorchTensorSample)
def _(y_pred: TorchTensorSample, y_true: torch.Tensor) -> torch.Tensor:
    """CQR-r nonconformity scores for TorchTensorSamples."""
    return cqr_r_score_func(y_pred.tensor, y_true)


@weight_func.register(torch.Tensor)
def _(y_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Weights for CQR-r score normalization for PyTorch tensors.

    CQR-r calibrates in width-normalized score space (s = residual/width), so
    the test-time adjustment must multiply back by width: [lower - q*width, upper + q*width].
    """
    if y_pred.ndim != 2 or y_pred.shape[1] != 2:
        msg = f"y_pred must have shape (n_samples, 2), got {y_pred.shape}"
        raise ValueError(msg)

    width = torch.clamp(y_pred[:, 1] - y_pred[:, 0], min=_EPS)
    return width, width


@weight_func.register(TorchTensorSample)
def _(y_pred: TorchTensorSample) -> tuple[torch.Tensor, torch.Tensor]:
    """Weights for CQR-r score normalization for TorchTensorSamples."""
    return weight_func(y_pred.tensor)
