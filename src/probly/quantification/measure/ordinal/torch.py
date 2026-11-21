"""PyTorch implementations for ordinal classification measures."""

from __future__ import annotations

import torch

from probly.representation.distribution.torch_categorical import TorchCategoricalDistributionSample

from ._common import (
    LogBase,
    categorical_variance_aleatoric,
    categorical_variance_total,
)


@categorical_variance_total.register(TorchCategoricalDistributionSample)
def torch_categorical_sample_ordinal_integer_variance_total(
    sample: TorchCategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> torch.Tensor:
    """Compute the total variance under integer label encoding for a torch categorical sample."""
    p = sample.tensor.probabilities
    axis = sample.sample_axis
    labels = torch.arange(1, p.shape[-1] + 1, dtype=p.dtype, device=p.device)
    expected_p = torch.mean(p, dim=axis)
    mu = torch.sum(labels * expected_p, dim=-1, keepdim=True)
    return torch.sum(((labels - mu) ** 2) * expected_p, dim=-1)


@categorical_variance_aleatoric.register(TorchCategoricalDistributionSample)
def torch_categorical_sample_ordinal_integer_variance_aleatoric(
    sample: TorchCategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> torch.Tensor:
    """Compute the aleatoric variance under integer label encoding for a torch categorical sample."""
    p = sample.tensor.probabilities
    axis = sample.sample_axis
    labels = torch.arange(1, p.shape[-1] + 1, dtype=p.dtype, device=p.device)
    mu_m = torch.sum(labels * p, dim=-1, keepdim=True)
    per_model = torch.sum(((labels - mu_m) ** 2) * p, dim=-1)
    return torch.mean(per_model, dim=axis)
