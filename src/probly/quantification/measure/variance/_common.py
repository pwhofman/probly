from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from flextype import flexdispatch
from probly.representation.distribution import Distribution

if TYPE_CHECKING:
    from probly.representation.array_like import ArrayLike
    from probly.representation.distribution import SecondOrderDistribution
    from probly.representation.sample import Sample


type LogBase = float | Literal["normalize"] | None


@flexdispatch
def variance(distribution: Distribution, base: LogBase = None) -> ArrayLike:
    """Compute the variance of a distribution."""
    msg = f"Variance is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


type SecondOrderDistributionLike = SecondOrderDistribution | Sample[Distribution]


@flexdispatch
def variance_of_expected_predictive_distribution(
    distribution: SecondOrderDistributionLike, base: LogBase = None
) -> ArrayLike:
    """Compute the variance of the expected value of a second-order distribution."""
    msg = f"Variance of expected value is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def conditional_variance(distribution: SecondOrderDistributionLike, base: LogBase = None) -> ArrayLike:
    """Compute the conditional variance of a distribution."""
    msg = f"Conditional variance is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def mutual_information_variance(distribution: SecondOrderDistributionLike, base: LogBase = None) -> ArrayLike:
    """Compute the mutual information of a distribution."""
    msg = f"Mutual information is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)
