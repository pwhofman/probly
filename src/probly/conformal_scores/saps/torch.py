"""Torch implementation for SAPS scores."""

from __future__ import annotations

import torch

from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.representation.sample.torch import TorchSample

from ._common import _saps_score_dispatch


@_saps_score_dispatch.register(torch.Tensor)
def compute_saps_score_torch(
    probs: torch.Tensor,
    y_cal: torch.Tensor | None = None,
    randomized: bool = True,
    lambda_val: float = 0.1,
) -> torch.Tensor:
    """SAPS Nonconformity-Scores for PyTorch tensors."""
    probs_torch = torch.as_tensor(probs, dtype=torch.float)
    if probs_torch.ndim < 1:
        msg = (
            "probs must have at least one dimension with classes on the last axis, "
            f"got shape {tuple(probs_torch.shape)}."
        )
        raise ValueError(msg)

    u = torch.rand_like(probs_torch) if randomized else torch.zeros_like(probs_torch)

    max_probs = torch.max(probs_torch, dim=-1, keepdim=True).values
    sort_idx = torch.argsort(-probs_torch, dim=-1)
    ranks_zero_based = torch.argsort(sort_idx, dim=-1)
    ranks = ranks_zero_based + 1

    scores = torch.where(ranks == 1, u * max_probs, max_probs + (ranks - 2 + u) * lambda_val)

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


@_saps_score_dispatch.register(TorchSample)
def _(
    probs: TorchSample,
    y_cal: torch.Tensor | None = None,
    randomized: bool = True,
    lambda_val: float = 0.1,
) -> torch.Tensor:
    """SAPS Nonconformity-Scores for TorchSamples."""
    return _saps_score_dispatch(
        probs.tensor,
        y_cal,
        randomized=randomized,
        lambda_val=lambda_val,
    )


@_saps_score_dispatch.register(TorchCategoricalDistribution)
def _(
    probs: TorchCategoricalDistribution,
    y_cal: torch.Tensor | None = None,
    randomized: bool = True,
    lambda_val: float = 0.1,
) -> torch.Tensor:
    """SAPS Nonconformity-Scores for TorchCategoricalDistributions."""
    return compute_saps_score_torch(
        probs.probabilities,
        y_cal,
        randomized=randomized,
        lambda_val=lambda_val,
    )
