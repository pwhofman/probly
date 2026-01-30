"""Torch for SAPS."""

from __future__ import annotations

import torch

from .common import register


def saps_score_torch(
    probs: torch.Tensor,
    lambda_val: float,
    u: torch.Tensor,
) -> torch.Tensor:
    """Compute SAPS nonconformity score for torch tensors.

    Args:
        probs: 1D tensor with softmax probabilities.
        lambda_val: lambda value for SAPS.
        u: optional random value in [0,1).

    Returns:
        torch.Tensor: SAPS nonconformity score.
    """
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, device=probs.device, dtype=probs.dtype)

    # convert to torch tensors
    probs = torch.asarray(probs, dtype=torch.float)
    u = torch.asarray(u, dtype=torch.float)

    # get max probabilities for each sample
    max_probs = torch.max(probs, dim=1, keepdim=True).values

    # get ranks for each label, argsort along axis=1 in descending order
    sort_idx = torch.argsort(-probs, dim=1)

    # find the rank (1-based) of each label
    # compare each position in sorted_indices with the corresponding label
    ranks_zero_based = torch.argsort(sort_idx, dim=1)
    ranks = ranks_zero_based + 1  # +1 for 1-based rank

    scores = torch.where(ranks == 1, u * max_probs, max_probs + (ranks - 2 + u) * lambda_val)

    return torch.asarray(scores, dtype=torch.float)


register(torch.Tensor, saps_score_torch)
