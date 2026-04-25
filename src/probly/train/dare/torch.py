"""Collection of torch dare training functions."""

from __future__ import annotations

import torch
from torch import nn

EPS_LOG = torch.finfo(torch.float32).eps


def dare_regularizer(
    model: nn.Module,
    device: torch.device | str,
    loss: torch.Tensor,
    threshold: torch.Tensor | float,
) -> torch.Tensor:
    """Compute the DARE anti-regularization term following Algorithm 1.

    Args:
        model: The DARE model.
        device: The device of the model.
        loss: The current loss value, used for the switching condition.
        threshold: The threshold at or below which anti-regularization activates.

    Returns:
        The anti-regularization term when loss <= threshold, else 0.0.
    """
    if loss <= threshold:
        anti_reg = torch.zeros((), device=device)
        d = 0
        for param in model.parameters():
            if param.requires_grad:
                anti_reg = anti_reg + torch.sum(torch.log(param.pow(2) + EPS_LOG))
                d += param.numel()
        return anti_reg / d
    return torch.zeros((), device=device)
