"""Torch implementations of ordinal uncertainty decompositions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from probly.representation.distribution.torch_categorical import TorchCategoricalDistributionSample
from probly.representation.distribution.torch_gaussian import TorchGaussianDistributionSample

from ._common import (
    categorical_variance_aleatoric,
    categorical_variance_total,
    gaussian_variance_aleatoric,
    gaussian_variance_epistemic,
    labelwise_binary_entropy_aleatoric,
    labelwise_binary_entropy_total,
    labelwise_binary_variance_aleatoric,
    labelwise_binary_variance_total,
    ordinal_binary_entropy_aleatoric,
    ordinal_binary_entropy_total,
    ordinal_binary_variance_aleatoric,
    ordinal_binary_variance_total,
)

if TYPE_CHECKING:
    from probly.quantification.measure.distribution._common import LogBase


def _torch_binary_entropy(p: torch.Tensor, base: LogBase = None) -> torch.Tensor:
    """Compute the binary Shannon entropy of probabilities ``p``.

    Uses :func:`torch.special.xlogy` so that ``p = 0`` and ``p = 1`` are handled
    correctly (``0 log 0 = 0``). ``base="normalize"`` divides by ``log(2)`` so
    that the entropy is in ``[0, 1]`` for each binary problem.
    """
    q = 1.0 - p
    h = -torch.special.xlogy(p, p) - torch.special.xlogy(q, q)
    if base is None:
        return h
    if base == "normalize":
        divisor = torch.log(torch.as_tensor(2.0, dtype=h.dtype, device=h.device))
    else:
        divisor = torch.log(torch.as_tensor(float(base), dtype=h.dtype, device=h.device))
    return h / divisor


def _cumulative_lower(p: torch.Tensor) -> torch.Tensor:
    """Return ``p_<=k`` for ``k = 1, ..., K-1`` along the trailing class axis."""
    cum = torch.cumsum(p, dim=-1)
    return cum[..., :-1]


# OCS (Order-Consistent Split) binary reduction


@ordinal_binary_entropy_total.register(TorchCategoricalDistributionSample)
def torch_ordinal_binary_entropy_total(
    sample: TorchCategoricalDistributionSample, base: LogBase = None
) -> torch.Tensor:
    """OCS binary-entropy total uncertainty for a torch categorical sample."""
    p = sample.tensor.probabilities
    axis = sample.sample_axis
    del sample
    cum = _cumulative_lower(p)
    expected_cum = torch.mean(cum, dim=axis)
    binary_h = _torch_binary_entropy(expected_cum, base=base)
    return torch.sum(binary_h, dim=-1)


@ordinal_binary_entropy_aleatoric.register(TorchCategoricalDistributionSample)
def torch_ordinal_binary_entropy_aleatoric(
    sample: TorchCategoricalDistributionSample, base: LogBase = None
) -> torch.Tensor:
    """OCS binary-entropy aleatoric uncertainty for a torch categorical sample."""
    p = sample.tensor.probabilities
    axis = sample.sample_axis
    del sample
    cum = _cumulative_lower(p)
    binary_h = _torch_binary_entropy(cum, base=base)
    per_model_sum = torch.sum(binary_h, dim=-1)
    return torch.mean(per_model_sum, dim=axis)


@ordinal_binary_variance_total.register(TorchCategoricalDistributionSample)
def torch_ordinal_binary_variance_total(sample: TorchCategoricalDistributionSample) -> torch.Tensor:
    """OCS binary-variance total uncertainty for a torch categorical sample."""
    p = sample.tensor.probabilities
    axis = sample.sample_axis
    del sample
    cum = _cumulative_lower(p)
    expected_cum = torch.mean(cum, dim=axis)
    return torch.sum(expected_cum * (1.0 - expected_cum), dim=-1)


@ordinal_binary_variance_aleatoric.register(TorchCategoricalDistributionSample)
def torch_ordinal_binary_variance_aleatoric(sample: TorchCategoricalDistributionSample) -> torch.Tensor:
    """OCS binary-variance aleatoric uncertainty for a torch categorical sample."""
    p = sample.tensor.probabilities
    axis = sample.sample_axis
    del sample
    cum = _cumulative_lower(p)
    per_model = torch.sum(cum * (1.0 - cum), dim=-1)
    return torch.mean(per_model, dim=axis)


# Label-wise (one-vs-rest) binary reduction


@labelwise_binary_entropy_total.register(TorchCategoricalDistributionSample)
def torch_labelwise_binary_entropy_total(
    sample: TorchCategoricalDistributionSample, base: LogBase = None
) -> torch.Tensor:
    """Label-wise binary-entropy total uncertainty for a torch categorical sample."""
    p = sample.tensor.probabilities
    axis = sample.sample_axis
    del sample
    expected_p = torch.mean(p, dim=axis)
    binary_h = _torch_binary_entropy(expected_p, base=base)
    return torch.sum(binary_h, dim=-1)


@labelwise_binary_entropy_aleatoric.register(TorchCategoricalDistributionSample)
def torch_labelwise_binary_entropy_aleatoric(
    sample: TorchCategoricalDistributionSample, base: LogBase = None
) -> torch.Tensor:
    """Label-wise binary-entropy aleatoric uncertainty for a torch categorical sample."""
    p = sample.tensor.probabilities
    axis = sample.sample_axis
    del sample
    binary_h = _torch_binary_entropy(p, base=base)
    per_model_sum = torch.sum(binary_h, dim=-1)
    return torch.mean(per_model_sum, dim=axis)


@labelwise_binary_variance_total.register(TorchCategoricalDistributionSample)
def torch_labelwise_binary_variance_total(sample: TorchCategoricalDistributionSample) -> torch.Tensor:
    """Label-wise binary-variance total uncertainty for a torch categorical sample."""
    p = sample.tensor.probabilities
    axis = sample.sample_axis
    del sample
    expected_p = torch.mean(p, dim=axis)
    return torch.sum(expected_p * (1.0 - expected_p), dim=-1)


@labelwise_binary_variance_aleatoric.register(TorchCategoricalDistributionSample)
def torch_labelwise_binary_variance_aleatoric(sample: TorchCategoricalDistributionSample) -> torch.Tensor:
    """Label-wise binary-variance aleatoric uncertainty for a torch categorical sample."""
    p = sample.tensor.probabilities
    axis = sample.sample_axis
    del sample
    per_model = torch.sum(p * (1.0 - p), dim=-1)
    return torch.mean(per_model, dim=axis)


# Standard categorical variance via law of total variance


def _integer_labels(num_classes: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Integer encoding ``1, ..., K`` as a 1-D tensor."""
    return torch.arange(1, num_classes + 1, dtype=dtype, device=device)


@categorical_variance_total.register(TorchCategoricalDistributionSample)
def torch_categorical_variance_total(sample: TorchCategoricalDistributionSample) -> torch.Tensor:
    """Total variance under integer label encoding for a torch categorical sample."""
    p = sample.tensor.probabilities
    axis = sample.sample_axis
    del sample
    labels = _integer_labels(p.shape[-1], dtype=p.dtype, device=p.device)
    expected_p = torch.mean(p, dim=axis)
    mu = torch.sum(labels * expected_p, dim=-1, keepdim=True)
    return torch.sum(((labels - mu) ** 2) * expected_p, dim=-1)


@categorical_variance_aleatoric.register(TorchCategoricalDistributionSample)
def torch_categorical_variance_aleatoric(sample: TorchCategoricalDistributionSample) -> torch.Tensor:
    """Aleatoric variance under integer label encoding for a torch categorical sample."""
    p = sample.tensor.probabilities
    axis = sample.sample_axis
    del sample
    labels = _integer_labels(p.shape[-1], dtype=p.dtype, device=p.device)
    mu_m = torch.sum(labels * p, dim=-1, keepdim=True)
    per_model = torch.sum(((labels - mu_m) ** 2) * p, dim=-1)
    return torch.mean(per_model, dim=axis)


# Gaussian regression variance via law of total variance


@gaussian_variance_aleatoric.register(TorchGaussianDistributionSample)
def torch_gaussian_variance_aleatoric(sample: TorchGaussianDistributionSample) -> torch.Tensor:
    """Aleatoric variance for a torch Gaussian sample."""
    axis = sample.sample_axis
    return torch.mean(sample.tensor.var, dim=axis)


@gaussian_variance_epistemic.register(TorchGaussianDistributionSample)
def torch_gaussian_variance_epistemic(sample: TorchGaussianDistributionSample) -> torch.Tensor:
    """Epistemic variance (variance of the predicted means) for a torch Gaussian sample."""
    axis = sample.sample_axis
    return torch.var(sample.tensor.mean, dim=axis, correction=0)
