"""Torch implementation for RAPS scores."""

from __future__ import annotations

import torch

from .common import register


def raps_score_torch(
    probs: torch.Tensor,
    lambda_reg: float = 0.1,
    k_reg: int = 0,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """Compute RAPS scores for torch tensors."""
    n_samples, n_classes = probs.shape

    # sort indices in descending order
    srt_probs, srt_idx = torch.sort(probs, dim=1, descending=True)

    # calculate cumulative sums
    cumsum_probs = torch.cumsum(srt_probs, dim=1)

    # regularization penalty
    ranks = torch.arange(1, n_classes + 1, device=probs.device).reshape(1, -1)
    penalty = lambda_reg * torch.maximum(
        torch.zeros(1, device=probs.device),
        ranks - k_reg - 1,
    )

    # add epsilon for stability
    epsilon_penalty = epsilon * torch.ones((n_samples, n_classes), device=probs.device)

    # combine all components
    reg_cumsum = cumsum_probs + penalty + epsilon_penalty

    # sort scores back to original positions
    raps_scores = torch.zeros_like(probs)

    # distributes the cumsum values back according to the sorted_indices
    raps_scores.scatter_(1, srt_idx, reg_cumsum)
    return raps_scores


register(torch.Tensor, raps_score_torch)
