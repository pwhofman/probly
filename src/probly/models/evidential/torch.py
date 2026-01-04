"""Collection of torch evidential models."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Dirichlet


class NatPNLoss(nn.Module):
    """NatPN classification loss using a Dirichlet-Categorical Bayesian model."""

    def __init__(self, entropy_weight: float = 1e-4) -> None:
        """Initialize NatPN loss."""
        super().__init__()
        self.entropy_weight = entropy_weight

    def forward(self, alpha: Tensor, y: Tensor) -> Tensor:
        """Compute NatPN loss."""
        alpha0 = alpha.sum(dim=-1)
        idx = torch.arange(y.size(0), device=y.device)
        alpha_y = alpha[idx, y]

        expected_nll = torch.digamma(alpha0) - torch.digamma(alpha_y)
        entropy = Dirichlet(alpha).entropy()

        return (expected_nll - self.entropy_weight * entropy).mean()
