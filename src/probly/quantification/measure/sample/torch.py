"""Torch implementations of sample-based uncertainty measures."""

from __future__ import annotations

from typing import TYPE_CHECKING

from probly.quantification.measure.sample._common import mean_squared_distance_to_scaled_one_hot
from probly.representation.sample.torch import TorchSample
from probly.representation.torch_functions import torch_average

if TYPE_CHECKING:
    import torch


@mean_squared_distance_to_scaled_one_hot.register(TorchSample)
def torch_mean_squared_distance_to_scaled_one_hot(sample: TorchSample, scale: float | None = None) -> torch.Tensor:
    r"""Torch impl. Uses :math:`\|h_k - s e_c\|^2 = \|h_k\|^2 - 2s \max_j h_{k,j} + s^2` (no one-hot built)."""
    tensor = sample.tensor
    num_classes = tensor.shape[-1]
    target_scale = float(num_classes) if scale is None else float(scale)

    norm_sq = (tensor * tensor).sum(dim=-1)  # type: ignore[union-attr]
    max_logit = tensor.amax(dim=-1)
    per_member = norm_sq - 2.0 * target_scale * max_logit + target_scale * target_scale

    if sample.weights is not None:
        return torch_average(per_member, dim=sample.sample_dim, weights=sample.weights)
    return per_member.mean(dim=sample.sample_dim)
