"""Torch for APS."""

from __future__ import annotations

import torch

from .common import register


def aps_score_torch(probs: torch.Tensor) -> torch.Tensor:
    """Compute APS scores for torch tensors."""
    sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=1)
    ranks = torch.arange(1, probs.size(1) + 1, device=probs.device).float()
    aps_scores = torch.sum(cumsum_probs / ranks, dim=1)
    return aps_scores


register(torch.Tensor, aps_score_torch)
