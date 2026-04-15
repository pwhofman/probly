"""Torch for APS."""

from __future__ import annotations

import torch

from probly.representation.sample.torch import TorchSample

from ._common import aps_score_func


@aps_score_func.register(torch.Tensor)
def _(probs: torch.Tensor, y_cal: torch.Tensor | None = None, randomized: bool = True) -> torch.Tensor:
    """APS Nonconformity-Scores for PyTorch tensors."""
    probs_torch = torch.as_tensor(probs)

    # sorting indices for descending probabilities
    srt_idx = torch.argsort(-probs_torch, dim=1)

    # get sorted probabilities
    srt_probs = torch.gather(probs_torch, 1, srt_idx)

    # calculate cumulative sums
    cumsum_probs = torch.cumsum(srt_probs, dim=1)

    # sort back to original positions without in-place writes
    inv_idx = torch.argsort(srt_idx, dim=1)

    if randomized:
        u = torch.rand_like(probs_torch)
        cumsum_probs -= srt_probs * u

    scores = torch.gather(cumsum_probs, 1, inv_idx)
    if y_cal is not None:
        relevant_indices = torch.arange(probs_torch.shape[0]), y_cal
        scores = scores[relevant_indices]
    return scores


@aps_score_func.register(TorchSample)
def _(probs: TorchSample, y_cal: torch.Tensor | None = None, randomized: bool = True) -> torch.Tensor:
    """APS Nonconformity-Scores for PyTorch tensors."""
    return aps_score_func(probs.tensor, y_cal, randomized=randomized)
