"""Torch implementation for absolute error scores."""

from __future__ import annotations

import torch

from probly.representation.sample.torch import TorchSample

from ._common import absolute_error_score_func


@absolute_error_score_func.register(torch.Tensor)
def compute_absolute_error_score_func_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Absolute error for PyTorch tensors."""
    if y_pred.ndim > 2:
        msg = (
            "y_pred must have shape (n_evaluations, n_samples) or (n_samples,), "
            f"got {y_pred.shape}. The n_evaluations dimension is optional and "
            "will be averaged over if present."
        )
        raise ValueError(msg)
    if y_pred.ndim == 2:
        y_pred = y_pred.mean(dim=0)
    return torch.abs(y_true - y_pred)


@absolute_error_score_func.register(TorchSample)
def compute_absolute_error_score_func_torch_sample(y_pred: TorchSample, y_true: TorchSample) -> torch.Tensor:
    """Absolute error for TorchSamples."""
    return compute_absolute_error_score_func_torch(y_pred.tensor, y_true.tensor)
