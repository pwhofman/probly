"""Torch implementation of the Dirichlet distribution."""

from __future__ import annotations

import torch

from ._common import create_dirichlet_distribution_from_alphas


@create_dirichlet_distribution_from_alphas.register(torch.Tensor)
def _create_dirichlet_distribution_from_alphas(alphas: torch.Tensor) -> torch.distributions.Dirichlet:
    return torch.distributions.Dirichlet(alphas)
