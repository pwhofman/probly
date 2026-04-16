"""Torch implementation for SAPS scores."""

from __future__ import annotations

import torch

from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.representation.sample.torch import TorchSample

from ._common import saps_score_func


@saps_score_func.register(torch.Tensor)
def compute_saps_score_func_torch(
    probs: torch.Tensor,
    y_cal: torch.Tensor | None = None,
    randomized: bool = True,
    lambda_val: float = 0.1,
) -> torch.Tensor:
    """SAPS Nonconformity-Scores for PyTorch tensors."""
    probs = torch.as_tensor(probs, dtype=torch.float)
    n_samples, n_classes = probs.shape

    if randomized:
        u = torch.rand(n_samples, n_classes, device=probs.device, dtype=probs.dtype)
    else:
        u = torch.zeros(n_samples, n_classes, device=probs.device, dtype=probs.dtype)

    max_probs = torch.max(probs, dim=1, keepdim=True).values
    sort_idx = torch.argsort(-probs, dim=1)
    ranks_zero_based = torch.argsort(sort_idx, dim=1)
    ranks = ranks_zero_based + 1

    scores = torch.where(ranks == 1, u * max_probs, max_probs + (ranks - 2 + u) * lambda_val)

    if y_cal is not None:
        scores = scores[torch.arange(n_samples, device=probs.device), y_cal]
    return scores


@saps_score_func.register(TorchSample)
def _(
    probs: TorchSample,
    y_cal: torch.Tensor | None = None,
    randomized: bool = True,
    lambda_val: float = 0.1,
) -> torch.Tensor:
    """SAPS Nonconformity-Scores for TorchSamples."""
    return compute_saps_score_func_torch(
        probs.samples,
        y_cal,
        randomized=randomized,
        lambda_val=lambda_val,
    )


@saps_score_func.register(TorchCategoricalDistribution)
def _(
    probs: TorchCategoricalDistribution,
    y_cal: torch.Tensor | None = None,
    randomized: bool = True,
    lambda_val: float = 0.1,
) -> torch.Tensor:
    """SAPS Nonconformity-Scores for TorchCategoricalDistributions."""
    return compute_saps_score_func_torch(
        probs.probabilities,
        y_cal,
        randomized=randomized,
        lambda_val=lambda_val,
    )
