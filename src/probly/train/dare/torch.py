"""Collection of torch dare training functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from probly.method.dare.torch import DareModule, DarePredictor

from torch import nn


class RegularizerDare(nn.Module):
    """Implementation of the DARE anti-regularization term."""

    def __init__(self, lambda_reg: float = 0.01) -> None:
        """Initialize the DARE regularizer.

        Args:
            lambda_reg: The anti-regularization coefficient.
        """
        super().__init__()
        self.lambda_reg: float = lambda_reg

    def forward(self, model: DareModule) -> torch.Tensor:
        """Calculates the anti-regularization loss for the dare."""
        # Get device
        params = list(model.parameters())
        device = params[0].device if params else torch.device("cpu")

        reg_loss: torch.Tensor = torch.tensor(0.0, device=device)
        num_members = len(model)
        if num_members < 2:
            return reg_loss

        # Pre-flatten parameters for each member to speed up distance calculation
        flat_params = [torch.cat([p.view(-1) for p in model[i].parameters()]) for i in range(num_members)]

        # Calculate pairwise anti-regularization between members (Eq. 3)
        for i in range(num_members):
            for j in range(i + 1, num_members):
                dist_sq = torch.norm(flat_params[i] - flat_params[j], p=2).pow(2)
                # Use a small epsilon to avoid log(0)
                reg_loss -= torch.log(dist_sq + 1e-6)

        # Normalize loss by the number of pairs and scale by the lambda_reg coefficent.
        num_pairs = num_members * (num_members - 1) / 2
        return (reg_loss / num_pairs) * self.lambda_reg


def dare_loss_step(
    model: DarePredictor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    base_loss_fn: nn.Module,
    regularizer: RegularizerDare,
) -> torch.Tensor:
    """Perform a DARE training loss calculation following Algorithm 1.

    Args:
        model: The DARE model.
        inputs: Input data.
        targets: Target labels.
        base_loss_fn: The base loss function (e.g., CrossEntropyLoss).
        regularizer: The DARE regularizer.

    Returns:
        The total loss for the step.
    """
    # Forward pass: outputs has shape [num_members, batch_size, ...]
    outputs = model(inputs)
    num_members = len(model)

    # L(Theta) = 1/K * sum L(theta_k) (Eq. 1)
    member_losses = [base_loss_fn(outputs[i], targets) for i in range(num_members)]

    avg_loss = torch.stack(member_losses).mean()

    # DARE switching logic (Algorithm 1)
    if avg_loss < model.threshold:
        reg_loss = regularizer(model)
        total_loss = avg_loss + reg_loss
    else:
        total_loss = avg_loss

    return total_loss
