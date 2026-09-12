"""Torch implementation for Total Variation scores."""

from __future__ import annotations

import torch

from probly.representation.distribution import CategoricalDistribution
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.representation.sample.torch import TorchSample

from ._common import tv_score_func


@tv_score_func.register(torch.Tensor)
def compute_tv_score_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Computes the Total Variation score using Torch Tensor."""
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
        batch_shape = torch.broadcast_shapes(y_pred_t.shape[:-1], y_true_t.shape)
        probabilities = y_pred_t.expand(*batch_shape, y_pred_t.shape[-1])
        labels = y_true_t.expand(batch_shape).long()
        selected = torch.gather(probabilities, -1, labels.unsqueeze(-1)).squeeze(-1)
        # Replace the selected class's contribution without allocating a one-hot target.
        return 0.5 * (y_pred_t.abs().sum(dim=-1) - selected.abs() + (selected - 1.0).abs())

    if not distribution_target and not y_true_t.is_floating_point():
        msg = "Targets must be integer labels, floating-point probabilities, or a categorical distribution."
        raise TypeError(msg)
    if y_true_t.ndim == 0 or y_true_t.shape[-1] != y_pred_t.shape[-1]:
        msg = "Target probabilities must have the same number of classes as predictions."
        raise ValueError(msg)

    return 0.5 * torch.sum(torch.abs(y_pred_t - y_true_t), dim=-1)


@tv_score_func.register(TorchSample)
def _(y_pred: TorchSample, y_true: torch.Tensor) -> torch.Tensor:
    """Compute total variation scores for Torch samples."""
    return tv_score_func(y_pred.tensor, y_true)


@tv_score_func.register(TorchCategoricalDistribution)
def _(y_pred: TorchCategoricalDistribution, y_true: torch.Tensor) -> torch.Tensor:
    """Compute total variation scores for Torch categorical distributions."""
    return tv_score_func(y_pred.probabilities, y_true)
