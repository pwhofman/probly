"""Torch implementation for absolute error scores."""

from __future__ import annotations

import torch

from probly.representation.sample.torch import TorchSample

from ._common import absolute_error_score_func


@absolute_error_score_func.register(torch.Tensor)
def _(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Absolute error for PyTorch tensors."""
    return torch.abs(y_true - y_pred)


@absolute_error_score_func.register(TorchSample)
def _(y_pred: TorchSample, y_true: TorchSample) -> torch.Tensor:
    """Absolute error for TorchSamples."""
    return absolute_error_score_func(y_pred.tensor, y_true.tensor)
