"""Torch implementation for RAPS scores."""

from __future__ import annotations

import torch

from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.representation.sample.torch import TorchSample

from ._common import _raps_score_dispatch


@_raps_score_dispatch.register(torch.Tensor)
def compute_raps_score_torch(
    probs: torch.Tensor,
    y_cal: torch.Tensor | None = None,
    randomized: bool = True,
    lambda_reg: float = 0.1,
    k_reg: int = 0,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """RAPS Nonconformity-Scores for PyTorch tensors."""
    probs_torch = torch.as_tensor(probs, dtype=torch.float)
    if probs_torch.ndim < 1:
        msg = (
            "probs must have at least one dimension with classes on the last axis, "
            f"got shape {tuple(probs_torch.shape)}."
        )
        raise ValueError(msg)

    n_classes = probs_torch.shape[-1]

    # sorting indices for descending probabilities
    srt_idx = torch.argsort(-probs_torch, dim=-1)
    srt_probs = torch.gather(probs_torch, -1, srt_idx)

    # calculate cumulative sums
    cumsum_probs = torch.cumsum(srt_probs, dim=-1)

    if randomized:
        u = torch.rand_like(probs_torch)
        cumsum_probs -= srt_probs * u

    # regularization penalty
    ranks = torch.arange(1, n_classes + 1, device=probs_torch.device, dtype=probs_torch.dtype)
    ranks = ranks.view((1,) * (probs_torch.ndim - 1) + (-1,))
    penalty = lambda_reg * torch.clamp(ranks - k_reg - 1, min=0)
    epsilon_penalty = epsilon * torch.ones_like(probs_torch)

    reg_cumsum = cumsum_probs + penalty + epsilon_penalty

    # sort back to original positions
    inv_idx = torch.argsort(srt_idx, dim=-1)
    scores = torch.gather(reg_cumsum, -1, inv_idx)

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


@_raps_score_dispatch.register(TorchSample)
def _(
    probs: TorchSample,
    y_cal: torch.Tensor | None = None,
    randomized: bool = True,
    lambda_reg: float = 0.1,
    k_reg: int = 0,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """RAPS Nonconformity-Scores for TorchSamples."""
    return _raps_score_dispatch(
        probs.tensor,
        y_cal,
        randomized=randomized,
        lambda_reg=lambda_reg,
        k_reg=k_reg,
        epsilon=epsilon,
    )


@_raps_score_dispatch.register(TorchCategoricalDistribution)
def _(
    probs: TorchCategoricalDistribution,
    y_cal: torch.Tensor | None = None,
    randomized: bool = True,
    lambda_reg: float = 0.1,
    k_reg: int = 0,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """RAPS Nonconformity-Scores for TorchCategoricalDistributions."""
    return compute_raps_score_torch(
        probs.probabilities,
        y_cal,
        randomized=randomized,
        lambda_reg=lambda_reg,
        k_reg=k_reg,
        epsilon=epsilon,
    )
