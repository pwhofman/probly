"""Entropy measures for PyTorch tensor distributions."""

from __future__ import annotations

import torch

from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistribution,
    TorchCategoricalDistributionSample,
)
from probly.utils.torch import torch_entropy

from ._common import (
    LogBase,
    conditional_entropy,
    entropy,
    entropy_of_expected_value,
    expected_max_probability_complement,
    max_disagreement,
    max_probability_complement_of_expected,
    mutual_information,
)

# Entropy


@entropy.register(TorchCategoricalDistribution)
def torch_categorical_entropy(distribution: TorchCategoricalDistribution, base: LogBase = None) -> torch.Tensor:
    """Compute the entropy of a categorical distribution represented as a PyTorch tensor."""
    if isinstance(distribution, TorchCategoricalDistribution):
        p = distribution.probabilities
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        p = distribution
    entropy = torch_entropy(p)
    if base is None or base == torch.e:
        return entropy
    if base == "normalize":
        base = float(p.shape[-1])

    return entropy / torch.log(torch.tensor(base))


# Entropy of expected value


@entropy_of_expected_value.register(TorchCategoricalDistributionSample)
def torch_categorical_sample_entropy_of_expected_value(
    sample: TorchCategoricalDistributionSample, base: LogBase = None
) -> torch.Tensor:
    """Compute the entropy of the expected value of a sample from a categorical distribution."""
    p = sample.tensor.probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    expected_value = torch.mean(p, dim=axis)
    return torch_categorical_entropy(expected_value, base=base)


# Conditional entropy


@conditional_entropy.register(TorchCategoricalDistributionSample)
def torch_categorical_sample_conditional_entropy(
    sample: TorchCategoricalDistributionSample, base: LogBase = None
) -> torch.Tensor:
    """Compute the conditional entropy of a sample from a categorical distribution."""
    p = sample.tensor.probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    entropies = torch_categorical_entropy(p, base=base)
    return torch.mean(entropies, dim=axis)


# Mutual information


@mutual_information.register(TorchCategoricalDistributionSample)
def torch_categorical_sample_mutual_information(
    sample: TorchCategoricalDistributionSample, base: LogBase = None
) -> torch.Tensor:
    """Compute the mutual information of a sample from a categorical distribution."""
    p = sample.tensor.probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    expected_value_entropy = torch_categorical_entropy(torch.mean(p, dim=axis), base=base)
    conditional_entropy_value = torch.mean(torch_categorical_entropy(p, base=base), dim=axis)
    return expected_value_entropy - conditional_entropy_value


# Zero-one proper scoring rule measures


@max_probability_complement_of_expected.register(TorchCategoricalDistributionSample)
def torch_categorical_sample_max_probability_complement_of_expected(
    sample: TorchCategoricalDistributionSample,
) -> torch.Tensor:
    """Compute one minus the max probability of the expected value of a categorical sample."""
    p = sample.tensor.probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    expected_value = torch.mean(p, dim=axis)
    return 1.0 - torch.max(expected_value, dim=-1).values


@expected_max_probability_complement.register(TorchCategoricalDistributionSample)
def torch_categorical_sample_expected_max_probability_complement(
    sample: TorchCategoricalDistributionSample,
) -> torch.Tensor:
    """Compute the expected value of one minus the max probability of a categorical sample."""
    p = sample.tensor.probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    per_sample_complement = 1.0 - torch.max(p, dim=-1).values
    return torch.mean(per_sample_complement, dim=axis)


@max_disagreement.register(TorchCategoricalDistributionSample)
def torch_categorical_sample_max_disagreement(
    sample: TorchCategoricalDistributionSample,
) -> torch.Tensor:
    """Compute the expected gap between each sample's max probability and its probability on the BMA argmax."""
    p = sample.tensor.probabilities
    axis = sample.sample_axis
    del sample  # Avoid keeping a reference to the sample for memory efficiency
    expected_value = torch.mean(p, dim=axis, keepdim=True)
    bma_argmax = torch.argmax(expected_value, dim=-1, keepdim=True)
    per_sample_bma_prob = torch.take_along_dim(p, bma_argmax, dim=-1).squeeze(-1)
    per_sample_max = torch.max(p, dim=-1).values
    return torch.mean(per_sample_max - per_sample_bma_prob, dim=axis)
