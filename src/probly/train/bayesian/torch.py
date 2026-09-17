"""Collection of torch Bayesian training functions."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ELBOLoss(nn.Module):
    """Evidence lower bound loss based on :cite:`blundellWeightUncertainty2015`.

    Attributes:
        kl_penalty: float, weight for KL divergence term
    """

    def __init__(self, kl_penalty: float = 1e-5) -> None:
        """Initializes an instance of the ELBOLoss class.

        Args:
        kl_penalty: float, weight for KL divergence term
        """
        super().__init__()
        self.kl_penalty = kl_penalty

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, kl: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ELBO loss.

        Args:
            inputs: torch.Tensor of size (n_instances, n_classes)
            targets: torch.Tensor of size (n_instances,)
            kl: torch.Tensor, KL divergence of the model
        Returns:
            loss: torch.Tensor, mean loss value
        """
        loss = F.cross_entropy(inputs, targets) + self.kl_penalty * kl
        return loss
