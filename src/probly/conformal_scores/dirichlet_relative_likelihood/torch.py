"""Torch implementation for Dirichlet relative likelihood scores."""

from __future__ import annotations

import torch

from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution

from ._common import dirichlet_rl_score_func


@dirichlet_rl_score_func.register(torch.Tensor)
def compute_dirichlet_rl_score_torch(alphas: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Compute the Dirichlet relative likelihood score using Torch Tensors.

    Args:
        alphas: Dirichlet concentration parameters, shape (..., K).
        y_true: Ground truth class labels, shape (...,).
    """
    alphas_t = torch.as_tensor(alphas)
    y_true_t = torch.as_tensor(y_true).long()
    alpha_y = torch.gather(alphas_t, dim=-1, index=y_true_t.unsqueeze(-1)).squeeze(-1)
    alpha_max = torch.amax(alphas_t, dim=-1)
    return 1.0 - alpha_y / alpha_max


@dirichlet_rl_score_func.register(TorchDirichletDistribution)
def compute_dirichlet_rl_score_torch_dirichlet(
    dirichlet: TorchDirichletDistribution, y_true: torch.Tensor
) -> torch.Tensor:
    """Compute the score from a TorchDirichletDistribution."""
    return compute_dirichlet_rl_score_torch(dirichlet.alphas, y_true)
