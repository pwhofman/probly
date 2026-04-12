"""Torch implementation for absolute error scores."""

from __future__ import annotations

import torch

from ._common import absolute_error_score_func


@absolute_error_score_func.register(torch.Tensor)
def _(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Absolute error for PyTorch tensors."""
    return torch.abs(y_true - y_pred)
