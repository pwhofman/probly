"""Torch implementation for Wasserstein distance scores."""

from __future__ import annotations

import torch

from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.representation.sample.torch import TorchSample

from ._common import wasserstein_distance_score_func


@wasserstein_distance_score_func.register(torch.Tensor)
def compute_wasserstein_distance_score_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Computes the Wasserstein distance score using Torch Tensor."""
    y_pred_t = torch.asarray(y_pred)
    y_true_t = torch.asarray(y_true)

    if y_true_t.ndim == 1 or (y_true_t.shape[0] == 1 and y_true_t.numel() == y_pred_t.shape[0]):
        y_one_hot = torch.zeros_like(y_pred_t)
        y_one_hot[torch.arange(len(y_true_t)), y_true_t.flatten().int()] = 1.0
        y_true_t = y_one_hot

    return torch.sum(torch.abs(torch.cumsum(y_pred_t, dim=-1) - torch.cumsum(y_true_t, dim=-1)), dim=-1)


@wasserstein_distance_score_func.register(TorchSample)
def _(y_pred: TorchSample, y_true: torch.Tensor) -> torch.Tensor:
    """Compute Wasserstein distance scores for Torch samples."""
    return wasserstein_distance_score_func(y_pred.tensor, y_true)


@wasserstein_distance_score_func.register(TorchCategoricalDistribution)
def _(y_pred: TorchCategoricalDistribution, y_true: torch.Tensor) -> torch.Tensor:
    """Compute Wasserstein distance scores for Torch categorical distributions."""
    return wasserstein_distance_score_func(y_pred.probabilities, y_true)
