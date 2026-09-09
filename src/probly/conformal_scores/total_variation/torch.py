"""Torch implementation for Total Variation scores."""

from __future__ import annotations

import torch

from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.representation.sample.torch import TorchSample

from ._common import tv_score_func


@tv_score_func.register(torch.Tensor)
def compute_tv_score_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Computes the Total Variation score using Torch Tensor."""
    y_pred_t = torch.asarray(y_pred)
    y_true_t = torch.asarray(y_true)

    if y_true_t.ndim == 1 or (y_true_t.shape[0] == 1 and y_true_t.numel() == y_pred_t.shape[0]):
        y_one_hot = torch.zeros_like(y_pred_t)
        y_one_hot[torch.arange(len(y_true_t)), y_true_t.flatten().int()] = 1.0
        y_true_t = y_one_hot

    return 0.5 * torch.sum(torch.abs(y_pred_t - y_true_t), dim=-1)


@tv_score_func.register(TorchSample)
def _(y_pred: TorchSample, y_true: torch.Tensor) -> torch.Tensor:
    """Compute total variation scores for Torch samples."""
    return tv_score_func(y_pred.tensor, y_true)


@tv_score_func.register(TorchCategoricalDistribution)
def _(y_pred: TorchCategoricalDistribution, y_true: torch.Tensor) -> torch.Tensor:
    """Compute total variation scores for Torch categorical distributions."""
    return tv_score_func(y_pred.probabilities, y_true)
