"""Entropy measures for PyTorch tensor distributions."""

from __future__ import annotations

import torch

from probly.representation.distribution.torch_categorical import (
    TorchCategoricalDistribution,
    TorchCategoricalDistributionSample,
)
from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution
from probly.representation.distribution.torch_gaussian import TorchGaussianDistribution
from probly.representation.distribution.torch_sparse_log_categorical import TorchSparseLogCategoricalDistribution
from probly.utils.torch import torch_entropy

from ._common import (
    DEFAULT_MEAN_FIELD_FACTOR,
    LogBase,
    conditional_entropy,
    dempster_shafer_uncertainty,
    entropy,
    entropy_of_expected_predictive_distribution,
    expected_max_probability_complement,
    max_disagreement,
    max_probability_complement_of_expected,
    mutual_information,
    vacuity,
)

# Entropy


@entropy.register
def torch_categorical_entropy(
    distribution: TorchCategoricalDistribution | torch.Tensor, base: LogBase = None
) -> torch.Tensor:
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


@entropy.register(TorchSparseLogCategoricalDistribution)
def torch_sparse_log_categorical_entropy(
    distribution: TorchSparseLogCategoricalDistribution, base: LogBase = None
) -> torch.Tensor:
    """Compute the entropy of a sparse log categorical distribution."""
    return torch_categorical_entropy(distribution.probabilities, base=base)


@entropy.register(TorchDirichletDistribution)
def torch_dirichlet_entropy(
    distribution: TorchDirichletDistribution | torch.Tensor, base: LogBase = None
) -> torch.Tensor:
    """Compute the differential entropy of a torch Dirichlet distribution."""
    if isinstance(distribution, TorchDirichletDistribution):
        alphas = distribution.alphas
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        alphas = distribution

    alpha_0 = torch.sum(alphas, dim=-1)
    num_classes = alphas.shape[-1]

    log_beta = torch.sum(torch.lgamma(alphas), dim=-1) - torch.lgamma(alpha_0)
    digamma_sum = (alpha_0 - num_classes) * torch.digamma(alpha_0)
    digamma_individual = torch.sum((alphas - 1) * torch.digamma(alphas), dim=-1)
    result = log_beta + digamma_sum - digamma_individual

    if base is None or base == torch.e:
        return result
    if base == "normalize":
        msg = "Entropy normalization is not supported for Dirichlet distributions."
        raise ValueError(msg)
    return result / torch.log(torch.as_tensor(base, dtype=result.dtype, device=result.device))


# Entropy of expected value


@entropy_of_expected_predictive_distribution.register(TorchDirichletDistribution)
def torch_dirichlet_entropy_of_expected_predictive_distribution(
    distribution: TorchDirichletDistribution | torch.Tensor, base: LogBase = None
) -> torch.Tensor:
    """Compute the entropy of the expected value of a torch Dirichlet distribution."""
    if isinstance(distribution, torch.Tensor):
        distribution = TorchDirichletDistribution(alphas=distribution)

    expected_distribution = distribution.mean
    return torch_categorical_entropy(expected_distribution, base=base)


@entropy_of_expected_predictive_distribution.register(TorchCategoricalDistributionSample)
def torch_categorical_sample_entropy_of_expected_predictive_distribution(
    sample: TorchCategoricalDistributionSample, base: LogBase = None
) -> torch.Tensor:
    """Compute the entropy of the expected value of a sample from a categorical distribution."""
    expected_distribution = sample.sample_mean()
    return torch_categorical_entropy(expected_distribution, base=base)


# Conditional entropy


@conditional_entropy.register(TorchDirichletDistribution)
def torch_dirichlet_conditional_entropy(
    distribution: TorchDirichletDistribution | torch.Tensor, base: LogBase = None
) -> torch.Tensor:
    """Compute the expected categorical entropy under a torch Dirichlet distribution."""
    if isinstance(distribution, TorchDirichletDistribution):
        alphas = distribution.alphas
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        alphas = distribution

    alpha_0 = torch.sum(alphas, dim=-1, keepdim=True)
    mean = alphas / alpha_0
    result = torch.digamma(alpha_0 + 1.0).squeeze(-1) - torch.sum(mean * torch.digamma(alphas + 1.0), dim=-1)

    if base is None or base == torch.e:
        return result
    if base == "normalize":
        msg = "Entropy normalization is not supported for Dirichlet distributions."
        raise ValueError(msg)
    return result / torch.log(torch.as_tensor(base, dtype=result.dtype, device=result.device))


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


@mutual_information.register(TorchDirichletDistribution)
def torch_dirichlet_mutual_information(
    distribution: TorchDirichletDistribution | torch.Tensor, base: LogBase = None
) -> torch.Tensor:
    """Compute mutual information of a torch Dirichlet distribution."""
    return torch_dirichlet_entropy_of_expected_predictive_distribution(
        distribution, base=base
    ) - torch_dirichlet_conditional_entropy(distribution, base=base)


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
    expected_distribution = sample.sample_mean()
    return 1.0 - torch.max(expected_distribution.probabilities, dim=-1).values


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


# Vacuity


@vacuity.register(TorchDirichletDistribution)
def torch_dirichlet_vacuity(distribution: TorchDirichletDistribution | torch.Tensor) -> torch.Tensor:
    """Compute the vacuity K / alpha_0 of a torch Dirichlet distribution."""
    if isinstance(distribution, TorchDirichletDistribution):
        alphas = distribution.alphas
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        alphas = distribution

    num_classes = alphas.shape[-1]
    alpha_0 = torch.sum(alphas, dim=-1)
    return torch.as_tensor(num_classes, dtype=alpha_0.dtype, device=alpha_0.device) / alpha_0


@max_probability_complement_of_expected.register(TorchDirichletDistribution)
def torch_dirichlet_max_probability_complement_of_expected(
    distribution: TorchDirichletDistribution | torch.Tensor,
) -> torch.Tensor:
    """Compute one minus the max probability of the mean of a torch Dirichlet distribution.

    Closed form: ``1 - max_c (alpha_c / alpha_0)``.
    """
    if isinstance(distribution, TorchDirichletDistribution):
        alphas = distribution.alphas
        del distribution  # Avoid keeping a reference to the distribution for memory efficiency
    else:
        alphas = distribution

    alpha_0 = torch.sum(alphas, dim=-1, keepdim=True)
    mean = alphas / alpha_0
    return 1.0 - torch.max(mean, dim=-1).values


# Dempster-Shafer uncertainty


@dempster_shafer_uncertainty.register(TorchGaussianDistribution)
def torch_gaussian_dempster_shafer_uncertainty(
    distribution: TorchGaussianDistribution,
    mean_field_factor: float = DEFAULT_MEAN_FIELD_FACTOR,
) -> torch.Tensor:
    """Compute the Dempster-Shafer uncertainty of a Gaussian over logits."""
    mean = distribution.mean
    var = distribution.var
    del distribution  # Avoid keeping a reference to the distribution for memory efficiency

    num_classes = mean.shape[-1]
    adjusted = mean / torch.sqrt(1.0 + mean_field_factor * var)
    num_classes_tensor = torch.as_tensor(num_classes, dtype=mean.dtype, device=mean.device)
    return num_classes_tensor / (num_classes_tensor + torch.sum(torch.exp(adjusted), dim=-1))
