"""Torch evidential classification implementation."""

from __future__ import annotations

import torch
from torch import nn

from probly.method.evidential.classification.common import register


class _AddOne(nn.Module):
    """Elementwise +1 module, used to turn softplus evidence into Dirichlet alpha."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


def append_activation_torch(obj: nn.Module) -> nn.Sequential:
    """Append Softplus + 1 so the model outputs Dirichlet alpha based on :cite:`sensoyEvidentialDeep2018`.

    Softplus ensures non-negative evidence; the +1 shift produces Dirichlet
    concentration parameters alpha suitable for the evidential losses in
    :mod:`probly.train.evidential.torch`.
    """
    return nn.Sequential(obj, nn.Softplus(), _AddOne())


register(nn.Module, append_activation_torch)
