"""Torch DARE implementation."""

from __future__ import annotations

import torch
from torch import nn

from ._common import dare_anti_regularization

EPS_LOG = torch.finfo(torch.float32).eps


@dare_anti_regularization.register(nn.Module)
def _torch_dare_anti_regularization(
    model: nn.Module,
    device: torch.device | str,
    loss: torch.Tensor,
    threshold: torch.Tensor | float,
) -> torch.Tensor:
    """Compute the DARE anti-regularization term of a torch model."""
    if loss <= threshold:
        anti_reg = torch.zeros((), device=device)
        d = 0
        for param in model.parameters():
            if param.requires_grad:
                anti_reg = anti_reg + torch.sum(torch.log(param.pow(2) + EPS_LOG))
                d += param.numel()
        return anti_reg / d
    return torch.zeros((), device=device)
