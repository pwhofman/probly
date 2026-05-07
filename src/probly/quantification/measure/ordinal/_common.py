from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from flextype import flexdispatch

if TYPE_CHECKING:
    from probly.representation.array_like import ArrayLike
    from probly.representation.distribution import Distribution, SecondOrderDistribution
    from probly.representation.sample import Sample


type LogBase = float | Literal["normalize"] | None

type SecondOrderDistributionLike = SecondOrderDistribution | Sample[Distribution]


@flexdispatch
def ordinal_variance(distribution: Distribution, base: LogBase = None) -> ArrayLike:
    """Compute the ordinal variance of a distribution."""
    msg = f"Ordinal variance is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_entropy(distribution: Distribution, base: LogBase = None) -> ArrayLike:
    """Compute the ordinal entropy of a distribution."""
    msg = f"Ordinal entropy is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_variance_of_expected_predictive_distribution(
    distribution: SecondOrderDistributionLike, base: LogBase = None
) -> ArrayLike:
    """Compute the variance of the expected value of a second-order distribution."""
    msg = f"Variance of expected value is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_conditional_variance(distribution: SecondOrderDistributionLike, base: LogBase = None) -> ArrayLike:
    """Compute the conditional variance of a distribution."""
    msg = f"Conditional variance is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_mutual_information_variance(distribution: SecondOrderDistributionLike, base: LogBase = None) -> ArrayLike:
    """Compute the mutual information of a distribution."""
    msg = f"Mutual information is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_entropy_of_expected_predictive_distribution(
    distribution: SecondOrderDistributionLike, base: LogBase = None
) -> ArrayLike:
    """Compute the entropy of the expected value of a second-order distribution."""
    msg = f"Entropy of expected value is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_mutual_information_entropy(distribution: SecondOrderDistributionLike, base: LogBase = None) -> ArrayLike:
    """Compute the mutual information of a distribution."""
    msg = f"Mutual information is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def ordinal_conditional_entropy(distribution: SecondOrderDistributionLike, base: LogBase = None) -> ArrayLike:
    """Compute the conditional entropy of a distribution."""
    msg = f"Conditional entropy is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def categorical_variance_total(distribution: SecondOrderDistributionLike, base: LogBase = None) -> ArrayLike:
    """Compute the total variance of an integer-encoded categorical sample."""
    msg = f"Ordinal integer variance total is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def categorical_variance_aleatoric(distribution: SecondOrderDistributionLike, base: LogBase = None) -> ArrayLike:
    """Compute the aleatoric variance of an integer-encoded categorical sample."""
    msg = f"Ordinal integer variance aleatoric is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_entropy(distribution: Distribution, base: LogBase = None) -> ArrayLike:
    """Compute the label-wise binary entropy of a distribution."""
    msg = f"Label-wise entropy is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_variance(distribution: Distribution, base: LogBase = None) -> ArrayLike:
    """Compute the label-wise binary variance of a distribution."""
    msg = f"Label-wise variance is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_entropy_of_expected_predictive_distribution(
    distribution: SecondOrderDistributionLike, base: LogBase = None
) -> ArrayLike:
    """Compute the label-wise binary entropy of the expected predictive distribution."""
    msg = f"Label-wise entropy of EPD is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_conditional_entropy(distribution: SecondOrderDistributionLike, base: LogBase = None) -> ArrayLike:
    """Compute the label-wise conditional entropy of a distribution."""
    msg = f"Label-wise conditional entropy is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_mutual_information_entropy(distribution: SecondOrderDistributionLike, base: LogBase = None) -> ArrayLike:
    """Compute the label-wise entropy-based mutual information of a distribution."""
    msg = f"Label-wise mutual information (entropy) is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_variance_of_expected_predictive_distribution(
    distribution: SecondOrderDistributionLike, base: LogBase = None
) -> ArrayLike:
    """Compute the label-wise binary variance of the expected predictive distribution."""
    msg = f"Label-wise variance of EPD is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_conditional_variance(distribution: SecondOrderDistributionLike, base: LogBase = None) -> ArrayLike:
    """Compute the label-wise conditional variance of a distribution."""
    msg = f"Label-wise conditional variance is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def labelwise_mutual_information_variance(distribution: SecondOrderDistributionLike, base: LogBase = None) -> ArrayLike:
    """Compute the label-wise variance-based mutual information of a distribution."""
    msg = f"Label-wise mutual information (variance) is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)
