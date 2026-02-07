"""Absolute Error Score implementation."""

from __future__ import annotations

import torch

from .common import register


def absolute_error_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Compute absolute error |y - y_hat|."""
    return torch.abs(y_true - y_pred)


register(torch.Tensor, absolute_error_torch)
