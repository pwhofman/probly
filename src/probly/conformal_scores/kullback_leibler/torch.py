"""Torch implementation for Kullback-Leibler divergence scores."""

from __future__ import annotations

import torch

from probly.representation.distribution import CategoricalDistribution
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.representation.sample.torch import TorchSample

from ._common import kl_divergence_score_func


@kl_divergence_score_func.register(torch.Tensor)
def compute_kl_divergence_score_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Computes the Kullback-Leibler divergence score using Torch Tensor."""
    y_pred_t = torch.as_tensor(y_pred)
    distribution_target = isinstance(y_true, CategoricalDistribution)
    y_true_t = torch.as_tensor(y_true.probabilities if distribution_target else y_true, device=y_pred_t.device)
    if y_pred_t.ndim == 0:
        msg = "Predicted probabilities must have a class axis."
        raise ValueError(msg)

    eps = 1e-12
    if (
        not distribution_target
        and y_true_t.dtype != torch.bool
        and not (y_true_t.is_floating_point() or y_true_t.is_complex())
    ):
        batch_shape = torch.broadcast_shapes(y_pred_t.shape[:-1], y_true_t.shape)
        probabilities = y_pred_t.expand(*batch_shape, y_pred_t.shape[-1])
        labels = y_true_t.expand(batch_shape).long()
        selected = torch.gather(probabilities, -1, labels.unsqueeze(-1)).squeeze(-1)
        return -torch.log(torch.clamp(selected, min=eps))

    if not distribution_target and not y_true_t.is_floating_point():
        msg = "Targets must be integer labels, floating-point probabilities, or a categorical distribution."
        raise TypeError(msg)
    if y_true_t.ndim == 0 or y_true_t.shape[-1] != y_pred_t.shape[-1]:
        msg = "Target probabilities must have the same number of classes as predictions."
        raise ValueError(msg)

    y_pred_safe = torch.clamp(y_pred_t, min=eps)
    y_true_safe = torch.clamp(y_true_t, min=eps)

    return torch.sum(y_true_t * torch.log(y_true_safe / y_pred_safe), dim=-1)


@kl_divergence_score_func.register(TorchSample)
def _(y_pred: TorchSample, y_true: torch.Tensor) -> torch.Tensor:
    """Compute memberwise scores for Torch samples."""
    return kl_divergence_score_func(y_pred.tensor, y_true)


@kl_divergence_score_func.register(TorchCategoricalDistribution)
def _(y_pred: TorchCategoricalDistribution, y_true: torch.Tensor) -> torch.Tensor:
    """Compute the score from normalized categorical probabilities."""
    return kl_divergence_score_func(y_pred.probabilities, y_true)
