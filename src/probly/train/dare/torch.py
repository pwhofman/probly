"""Collection of torch dare training functions."""

from __future__ import annotations

import torch
from torch import nn


class RegularizerDare(nn.Module):
    """Implementation of the DARE anti-regularization term."""

    def __init__(self, threshold: float) -> None:
        """Initialize the DARE regularizer.

        Args:
            threshold: The threshold for the anti-regularization.
        """
        super().__init__()
        self.threshold: float = threshold


def dare_loss_step(
    model: nn.Module,
    device: str,
    loss: float,
    regularizer: RegularizerDare,
) -> float:
    """Perform a DARE training loss calculation following Algorithm 1.

    Args:
        model: The DARE model.
        device: The device of the model.
        loss: The loss value for the epoch.
        regularizer: The DARE regularizer.

    Returns:
        The anti-regularized loss value for the epoch.
    """
    # DARE switching logic
    if loss < regularizer.threshold:
        anti_reg = torch.empty(1, device=device)
        d = 0
        for param in model.parameters():
            if param.requires_grad:
                anti_reg += torch.sum(torch.log(param.pow(2) + 1e-10))
                d += param.numel()
        total_loss = loss - (anti_reg.item() / d)
        return total_loss
    return loss
