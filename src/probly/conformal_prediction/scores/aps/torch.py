"""Torch for APS."""

from __future__ import annotations

import torch

from .common import register


def aps_score_torch(probs: torch.Tensor) -> torch.Tensor:
    """Compute APS scores for torch tensors (all labels)."""
    # sort indices in descending order
    srt_probs, srt_idx = torch.sort(probs, dim=1, descending=True)

    # calculate cumulative sums
    cumsum_probs = torch.cumsum(srt_probs, dim=1)

    # sort scores back to original positions
    aps_scores = torch.zeros_like(probs)

    # distributes the cumsum values back according to the sorted_indices
    aps_scores.scatter_(1, srt_idx, cumsum_probs)
    return aps_scores


register(torch.Tensor, aps_score_torch)
