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

    # get max probabilities for each sample
    max_probs = torch.max(probs, dim=1, keepdim=True).values

    # get ranks for each label, argsort along axis=1 in descending order
    sort_idx = torch.argsort(probs, dim=1, descending=True)

    # find the rank (1-based) of each label
    # compare each position in sorted_indices with the corresponding label
    ranks_zero_based = torch.argsort(sort_idx, dim=1)
    ranks = ranks_zero_based + 1  # (N, K) +1 for 1-based rank

    term_rank1 = u * max_probs
    term_rank_other = max_probs + (ranks - 2 + u) * lambda_val

    # compute scores based on ranks
    scores = torch.where(
        ranks == 1,
        term_rank1,
        term_rank_other,
    )

    return scores


register(torch.Tensor, saps_score_torch)
