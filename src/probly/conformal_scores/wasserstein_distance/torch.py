"""Torch implementation for Wasserstein distance scores."""

from __future__ import annotations

import torch

from probly.representation.distribution import CategoricalDistribution
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.representation.sample.torch import TorchSample

from ._common import wasserstein_distance_score_func


@wasserstein_distance_score_func.register(torch.Tensor)
def compute_wasserstein_distance_score_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Computes the Wasserstein distance score using Torch Tensor."""
    y_pred_t = torch.as_tensor(y_pred)
    distribution_target = isinstance(y_true, CategoricalDistribution)
    y_true_t = torch.as_tensor(y_true.probabilities if distribution_target else y_true, device=y_pred_t.device)
    if y_pred_t.ndim == 0:
        msg = "Predicted probabilities must have a class axis."
        raise ValueError(msg)

    if (
        not distribution_target
        and y_true_t.dtype != torch.bool
        and not (y_true_t.is_floating_point() or y_true_t.is_complex())
    ):
        # A point mass has a step-function CDF; no one-hot array or target cumsum is needed.
        target_cdf = torch.arange(y_pred_t.shape[-1], device=y_pred_t.device) >= y_true_t.unsqueeze(-1)
        cdf = torch.cumsum(y_pred_t, dim=-1)
        return torch.where(target_cdf, cdf - 1.0, cdf).abs().sum(dim=-1)

    if not distribution_target and not y_true_t.is_floating_point():
        msg = "Targets must be integer labels, floating-point probabilities, or a categorical distribution."
        raise TypeError(msg)
    if y_true_t.ndim == 0 or y_true_t.shape[-1] != y_pred_t.shape[-1]:
        msg = "Target probabilities must have the same number of classes as predictions."
        raise ValueError(msg)

    return torch.sum(torch.abs(torch.cumsum(y_pred_t, dim=-1) - torch.cumsum(y_true_t, dim=-1)), dim=-1)


@wasserstein_distance_score_func.register(TorchSample)
def _(y_pred: TorchSample, y_true: torch.Tensor) -> torch.Tensor:
    """Compute Wasserstein distance scores for Torch samples."""
    return wasserstein_distance_score_func(y_pred.tensor, y_true)


@wasserstein_distance_score_func.register(TorchCategoricalDistribution)
def _(y_pred: TorchCategoricalDistribution, y_true: torch.Tensor) -> torch.Tensor:
    """Compute Wasserstein distance scores for Torch categorical distributions."""
    return wasserstein_distance_score_func(y_pred.probabilities, y_true)
