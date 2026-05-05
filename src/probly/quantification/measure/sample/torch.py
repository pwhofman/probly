"""Torch implementations of sample-based uncertainty measures."""

from __future__ import annotations

import torch

from probly.quantification.measure.sample._common import (
    mean_squared_distance_to_scaled_one_hot,
    total_logit_sample_variance,
)
from probly.representation.distribution.torch_categorical import TorchCategoricalDistributionSample
from probly.representation.torch_functions import torch_average


@mean_squared_distance_to_scaled_one_hot.register(TorchCategoricalDistributionSample)
def torch_mean_squared_distance_to_scaled_one_hot(
    sample: TorchCategoricalDistributionSample, scale: float | None = None
) -> torch.Tensor:
    r"""Torch impl. Uses :math:`\|h_k - s e_c\|^2 = \|h_k\|^2 - 2s \max_j h_{k,j} + s^2` (no one-hot built)."""
    tensor = sample.tensor.logits
    num_classes = tensor.shape[-1]
    target_scale = float(num_classes) if scale is None else float(scale)

    norm_sq = (tensor * tensor).sum(dim=-1)  # type: ignore[union-attr]
    max_logit = tensor.amax(dim=-1)
    per_member = norm_sq - 2.0 * target_scale * max_logit + target_scale * target_scale

    if sample.weights is not None:
        return torch_average(per_member, dim=sample.sample_dim, weights=sample.weights)
    return per_member.mean(dim=sample.sample_dim)


@total_logit_sample_variance.register(TorchCategoricalDistributionSample)
def torch_total_logit_sample_variance(sample: TorchCategoricalDistributionSample) -> torch.Tensor:
    """Torch impl. Variance of total logits (logits summed across members)."""
    tensor = sample.tensor.logits
    return torch.var(tensor, dim=sample.sample_dim).sum(dim=-1)
