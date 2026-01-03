"""Torch for LAC."""

from __future__ import annotations

import torch

from .common import register


def lac_score_torch(probs: torch.Tensor) -> torch.Tensor:
    """Compute APS scores for torch tensors."""
    lac_scores = 1.0 - probs
    return lac_scores  # shape: (n_samples, n_classes)


register(torch.Tensor, lac_score_torch)
