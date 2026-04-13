"""PyTorch implementation of AUC."""

from __future__ import annotations

import torch

from probly.metrics import auc


@auc.register(torch.Tensor)
def auc_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute area under a curve using the trapezoid rule."""
    return torch.trapezoid(y, x, dim=-1)
