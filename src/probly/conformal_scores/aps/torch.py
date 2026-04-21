"""Torch for APS."""

from __future__ import annotations

import torch

from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.representation.sample.torch import TorchSample

from ._common import _aps_score_dispatch


@_aps_score_dispatch.register(torch.Tensor)
def compute_aps_score_torch(
    probs: torch.Tensor, y_cal: torch.Tensor | None = None, randomized: bool = True
) -> torch.Tensor:
    """APS Nonconformity-Scores for PyTorch tensors."""
    probs_torch = torch.as_tensor(probs, dtype=torch.float)

    if probs_torch.ndim < 1:
        msg = (
            "probs must have at least one dimension with classes on the last axis, "
            f"got shape {tuple(probs_torch.shape)}."
        )
        raise ValueError(msg)

    # sorting indices for descending probabilities
    srt_idx = torch.argsort(-probs_torch, dim=-1)

    # get sorted probabilities
    srt_probs = torch.gather(probs_torch, -1, srt_idx)

    # calculate cumulative sums
    cumsum_probs = torch.cumsum(srt_probs, dim=-1)

    # sort back to original positions without in-place writes
    inv_idx = torch.argsort(srt_idx, dim=-1)

    if randomized:
        u = torch.rand_like(probs_torch)
        cumsum_probs -= srt_probs * u

    scores = torch.gather(cumsum_probs, -1, inv_idx)
    if y_cal is not None:
        labels = torch.as_tensor(y_cal, device=probs_torch.device, dtype=torch.long)
        if tuple(labels.shape) != tuple(probs_torch.shape[:-1]):
            msg = (
                "y_cal must match probs batch shape (all axes except the class axis); "
                f"got y_cal shape {tuple(labels.shape)} and probs shape {tuple(probs_torch.shape)}."
            )
            raise ValueError(msg)
        scores = torch.gather(scores, -1, labels.unsqueeze(-1))
        scores = scores.squeeze(-1)
    return scores


@_aps_score_dispatch.register(TorchSample)
def _(probs: TorchSample, y_cal: torch.Tensor | None = None, randomized: bool = True) -> torch.Tensor:
    """APS Nonconformity-Scores for PyTorch tensors."""
    return _aps_score_dispatch(probs.tensor, y_cal, randomized=randomized)


@_aps_score_dispatch.register(TorchCategoricalDistribution)
def _(probs: TorchCategoricalDistribution, y_cal: torch.Tensor | None = None, randomized: bool = True) -> torch.Tensor:
    """APS Nonconformity-Scores for PyTorch tensors."""
    return compute_aps_score_torch(probs.probabilities, y_cal, randomized=randomized)
