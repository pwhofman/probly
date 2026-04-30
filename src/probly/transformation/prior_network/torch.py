"""Torch Prior Network implementation."""

from __future__ import annotations

import torch
from torch import nn

from probly.transformation.prior_network._common import register


class _Exp(nn.Module):
    """Elementwise exp module, turning logits into Dirichlet alpha per Malinin and Gales (2018)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)


def append_activation_torch(obj: nn.Module) -> nn.Sequential:
    """Append exp so the model outputs Dirichlet alpha based on :cite:`malininPredictiveUncertaintyEstimation2018`.

    Unlike Sensoy's softplus + 1 parameterization, exp allows alpha values below 1,
    which the Prior Networks paper uses for very flat Dirichlets on out-of-distribution inputs.
    """
    return nn.Sequential(obj, _Exp())


register(nn.Module, append_activation_torch)
