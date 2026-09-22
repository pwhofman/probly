"""Torch implementations of regression measures."""

from __future__ import annotations

import torch

from probly.representation.distribution.torch_gaussian import (
    TorchGaussianDistribution,
    TorchGaussianDistributionSample,
)
from probly.representation.sample.torch import TorchSample

from ._common import (
    LogBase,
    expected_conditional_variance,
    variance,
    variance_of_conditional_mean,
    variance_of_expected_predictive_distribution,
)


@variance.register(TorchGaussianDistribution)
def torch_gaussian_variance(distribution: TorchGaussianDistribution | torch.Tensor) -> torch.Tensor:
    """Compute the variance of a Gaussian distribution."""
    if isinstance(distribution, TorchGaussianDistribution):
        return distribution.var
    return distribution


@variance_of_expected_predictive_distribution.register(TorchGaussianDistributionSample)
def torch_gaussian_sample_variance_of_expected_predictive_distribution(
    sample: TorchGaussianDistributionSample, base: LogBase = None
) -> torch.Tensor:
    """Compute the total variance of the expected value of a second-order distribution."""
    aleatoric = torch_gaussian_sample_expected_conditional_variance(sample, base)
    epistemic = torch_gaussian_sample_variance_of_conditional_mean(sample, base)
    return aleatoric + epistemic


@expected_conditional_variance.register(TorchGaussianDistributionSample)
def torch_gaussian_sample_expected_conditional_variance(
    sample: TorchGaussianDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> torch.Tensor:
    """Compute the aleatoric variance of a Gaussian sample (mean of per-model variances)."""
    return torch.mean(sample.tensor.var, dim=sample.sample_axis)


@variance_of_conditional_mean.register(TorchGaussianDistributionSample)
def torch_gaussian_sample_variance_of_conditional_mean(
    sample: TorchGaussianDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> torch.Tensor:
    """Compute the epistemic variance of a Gaussian sample (variance of per-model means)."""
    return torch.var(sample.tensor.mean, dim=sample.sample_axis, unbiased=False)


@variance_of_expected_predictive_distribution.register(TorchSample)
def torch_sample_variance_of_expected_predictive_distribution(
    sample: TorchSample,
    base: LogBase = None,  # noqa: ARG001
) -> torch.Tensor:
    """Compute the total variance of the expected value of a raw tensor sample."""
    return torch.var(sample.tensor, dim=sample.sample_axis, unbiased=False)


@expected_conditional_variance.register(TorchSample)
def torch_sample_expected_conditional_variance(
    sample: TorchSample,
    base: LogBase = None,  # noqa: ARG001
) -> torch.Tensor:
    """Compute the expected conditional variance of a raw tensor sample."""
    return torch.zeros_like(torch.mean(sample.tensor, dim=sample.sample_axis))


@variance_of_conditional_mean.register(TorchSample)
def torch_sample_variance_of_conditional_mean(
    sample: TorchSample,
    base: LogBase = None,  # noqa: ARG001
) -> torch.Tensor:
    """Compute the epistemic variance of a raw tensor sample."""
    return torch.var(sample.tensor, dim=sample.sample_axis, unbiased=False)
