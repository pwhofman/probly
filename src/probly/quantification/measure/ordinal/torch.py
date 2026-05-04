"""PyTorch implementations for ordinal classification measures."""

from __future__ import annotations

import torch

from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistribution,
    TorchCategoricalDistributionSample,
)
from probly.utils.torch import torch_entropy

from ._common import (
    LogBase,
    ordinal_conditional_entropy,
    ordinal_conditional_variance,
    ordinal_entropy,
    ordinal_entropy_of_expected_predictive_distribution,
    ordinal_mutual_information_entropy,
    ordinal_mutual_information_variance,
    ordinal_variance,
    ordinal_variance_of_expected_predictive_distribution,
)


def _binary_entropy(p: torch.Tensor, base: LogBase = None) -> torch.Tensor:
    p_stack = torch.stack([p, 1 - p], dim=-1)
    entropy = torch_entropy(p_stack)
    if base is None or base == torch.e:
        return entropy
    if base == "normalize":
        base = 2.0

    return entropy / torch.log(torch.tensor(base, dtype=entropy.dtype, device=entropy.device))


def _cdf(p: torch.Tensor) -> torch.Tensor:
    """Compute the cumulative distribution function (CDF) excluding the last bin."""
    return torch.cumsum(p, dim=-1)[..., :-1]


@ordinal_variance.register(TorchCategoricalDistribution)
def torch_categorical_ordinal_variance(
    distribution: TorchCategoricalDistribution | torch.Tensor,
    base: LogBase = None,  # noqa: ARG001
) -> torch.Tensor:
    """Compute the ordinal variance of a categorical distribution."""
    if isinstance(distribution, TorchCategoricalDistribution):
        p = distribution.probabilities
        del distribution
    else:
        p = distribution
    cdf = _cdf(p)
    return torch.sum(cdf * (1 - cdf), dim=-1)


@ordinal_entropy.register(TorchCategoricalDistribution)
def torch_categorical_ordinal_entropy(
    distribution: TorchCategoricalDistribution | torch.Tensor, base: LogBase = None
) -> torch.Tensor:
    """Compute the ordinal entropy of a categorical distribution."""
    if isinstance(distribution, TorchCategoricalDistribution):
        p = distribution.probabilities
        del distribution
    else:
        p = distribution
    cdf = _cdf(p)
    binary_entropies = _binary_entropy(cdf, base=base)
    return torch.sum(binary_entropies, dim=-1)


@ordinal_variance_of_expected_predictive_distribution.register(TorchCategoricalDistributionSample)
def torch_categorical_sample_ordinal_variance_of_expected_predictive_distribution(
    sample: TorchCategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> torch.Tensor:
    """Compute the ordinal variance of the expected value of a categorical sample."""
    p = sample.tensor.probabilities
    axis = sample.sample_axis
    cdf = _cdf(p)
    expected_cdf = torch.mean(cdf, dim=axis)
    return torch.sum(expected_cdf * (1 - expected_cdf), dim=-1)


@ordinal_conditional_variance.register(TorchCategoricalDistributionSample)
def torch_categorical_sample_ordinal_conditional_variance(
    sample: TorchCategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> torch.Tensor:
    """Compute the ordinal conditional variance of a categorical sample."""
    p = sample.tensor.probabilities
    axis = sample.sample_axis
    cdf = _cdf(p)
    per_sample_variance = torch.sum(cdf * (1 - cdf), dim=-1)
    return torch.mean(per_sample_variance, dim=axis)


@ordinal_mutual_information_variance.register(TorchCategoricalDistributionSample)
def torch_categorical_sample_ordinal_mutual_information_variance(
    sample: TorchCategoricalDistributionSample,
    base: LogBase = None,  # noqa: ARG001
) -> torch.Tensor:
    """Compute the ordinal mutual information (variance-based) of a categorical sample."""
    p = sample.tensor.probabilities
    axis = sample.sample_axis
    cdf = _cdf(p)
    return torch.sum(torch.var(cdf, dim=axis, unbiased=False), dim=-1)


@ordinal_entropy_of_expected_predictive_distribution.register(TorchCategoricalDistributionSample)
def torch_categorical_sample_ordinal_entropy_of_expected_predictive_distribution(
    sample: TorchCategoricalDistributionSample, base: LogBase = None
) -> torch.Tensor:
    """Compute the ordinal entropy of the expected value of a categorical sample."""
    p = sample.tensor.probabilities
    axis = sample.sample_axis
    cdf = _cdf(p)
    expected_cdf = torch.mean(cdf, dim=axis)
    binary_entropies = _binary_entropy(expected_cdf, base=base)
    return torch.sum(binary_entropies, dim=-1)


@ordinal_conditional_entropy.register(TorchCategoricalDistributionSample)
def torch_categorical_sample_ordinal_conditional_entropy(
    sample: TorchCategoricalDistributionSample, base: LogBase = None
) -> torch.Tensor:
    """Compute the ordinal conditional entropy of a categorical sample."""
    p = sample.tensor.probabilities
    axis = sample.sample_axis
    cdf = _cdf(p)
    binary_entropies = _binary_entropy(cdf, base=base)
    per_sample_entropy = torch.sum(binary_entropies, dim=-1)
    return torch.mean(per_sample_entropy, dim=axis)


@ordinal_mutual_information_entropy.register(TorchCategoricalDistributionSample)
def torch_categorical_sample_ordinal_mutual_information_entropy(
    sample: TorchCategoricalDistributionSample, base: LogBase = None
) -> torch.Tensor:
    """Compute the ordinal mutual information (entropy-based) of a categorical sample."""
    return torch_categorical_sample_ordinal_entropy_of_expected_predictive_distribution(
        sample, base
    ) - torch_categorical_sample_ordinal_conditional_entropy(sample, base)
