"""Torch implementation for CQR-r scores."""

from __future__ import annotations

import torch

from probly.representation.sample.torch import TorchSample

from ._common import _EPS, cqr_r_score


@cqr_r_score.register(torch.Tensor)
def compute_cqr_r_score_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """CQR-r nonconformity scores for PyTorch tensors."""
    y = y_true.to(dtype=torch.float)
    pred = y_pred.to(dtype=torch.float)

    if pred.ndim < y.ndim + 1 or pred.shape[-1] != 2:
        msg = (
            "y_pred must have shape (..., 2) or (n_evaluations, ..., 2) matching y_true batch shape; "
            f"got y_pred shape {tuple(pred.shape)} and y_true shape {tuple(y.shape)}."
        )
        raise ValueError(msg)
    if pred.ndim == y.ndim + 2:
        pred = pred.mean(dim=0)
    elif pred.ndim != y.ndim + 1:
        msg = (
            "y_pred must match y_true batch rank with a trailing quantile axis, "
            "or include one additional leading evaluation axis."
        )
        raise ValueError(msg)

    lower = pred[..., 0]
    upper = pred[..., 1]
    width = torch.clamp(upper - lower, min=_EPS)

    return torch.maximum(lower - y, y - upper) / width


@cqr_r_score.register(TorchSample)
def _(y_pred: TorchSample, y_true: torch.Tensor) -> torch.Tensor:
    """CQR-r nonconformity scores for TorchSamples."""
    return compute_cqr_r_score_torch(y_pred.tensor, y_true)
