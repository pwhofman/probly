"""Torch implementation for absolute error scores."""

from __future__ import annotations

import torch

from probly.representation.sample.torch import TorchSample

from ._common import absolute_error_score


@absolute_error_score.register(torch.Tensor)
def compute_absolute_error_score_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Absolute error for PyTorch tensors."""
    y_pred_t = torch.as_tensor(y_pred, dtype=torch.float)
    y_true_t = torch.as_tensor(y_true, dtype=torch.float)

    if y_pred_t.ndim == y_true_t.ndim + 1:
        y_pred_t = y_pred_t.mean(dim=0)
    elif y_pred_t.ndim != y_true_t.ndim:
        msg = (
            "y_pred must match y_true shape or add a leading evaluation axis; "
            f"got y_pred shape {tuple(y_pred_t.shape)} and y_true shape {tuple(y_true_t.shape)}."
        )
        raise ValueError(msg)

    return torch.abs(y_true_t - y_pred_t)


@absolute_error_score.register(TorchSample)
def compute_absolute_error_score_torch_sample(
    y_pred: TorchSample,
    y_true: torch.Tensor | TorchSample,
) -> torch.Tensor:
    """Absolute error for TorchSamples."""
    y_true_t = y_true.tensor if isinstance(y_true, TorchSample) else y_true
    return compute_absolute_error_score_torch(y_pred.tensor, y_true_t)
