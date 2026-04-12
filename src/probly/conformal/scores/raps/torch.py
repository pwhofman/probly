"""Torch implementation for RAPS scores."""

from __future__ import annotations

import torch

from ._common import raps_score_func


@raps_score_func.register(torch.Tensor)
def _(
    probs: torch.Tensor,
    y_cal: torch.Tensor | None = None,
    randomized: bool = True,
    lambda_reg: float = 0.1,
    k_reg: int = 0,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """RAPS Nonconformity-Scores for PyTorch tensors."""
    n_samples, n_classes = probs.shape

    # sorting indices for descending probabilities
    srt_idx = torch.argsort(-probs, dim=1)
    srt_probs = torch.gather(probs, 1, srt_idx)

    # calculate cumulative sums
    cumsum_probs = torch.cumsum(srt_probs, dim=1)

    if randomized:
        U = torch.rand_like(probs)
        cumsum_probs -= srt_probs * U

    # regularization penalty
    ranks = torch.arange(1, n_classes + 1, device=probs.device).reshape(1, -1)
    penalty = lambda_reg * torch.clamp(ranks - k_reg - 1, min=0).to(probs.dtype)
    epsilon_penalty = epsilon * torch.ones((n_samples, n_classes), device=probs.device, dtype=probs.dtype)

    reg_cumsum = cumsum_probs + penalty + epsilon_penalty

    # sort back to original positions
    inv_idx = torch.argsort(srt_idx, dim=1)
    scores = torch.gather(reg_cumsum, 1, inv_idx)

    if y_cal is not None:
        scores = scores[torch.arange(n_samples, device=probs.device), y_cal]
    return scores
