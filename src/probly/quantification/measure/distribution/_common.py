"""Common definitions of distribution measures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from flextype import flexdispatch
from probly.representation.sample._common import Sample

if TYPE_CHECKING:
    from probly.representation.array_like import ArrayLike
    from probly.representation.distribution import Distribution, SecondOrderDistribution

type LogBase = float | Literal["normalize"] | None


@flexdispatch
def entropy(distribution: Distribution, base: LogBase = None) -> ArrayLike:
    """Compute the entropy of a distribution."""
    msg = f"Entropy is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


type SecondOrderDistributionLike = SecondOrderDistribution | Sample[Distribution]


@flexdispatch
def entropy_of_expected_value(distribution: SecondOrderDistributionLike, base: LogBase = None) -> ArrayLike:
    """Compute the entropy of the expected value of a second-order distribution."""
    msg = f"Entropy of expected value is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def conditional_entropy(distribution: SecondOrderDistributionLike, base: LogBase = None) -> ArrayLike:
    """Compute the conditional entropy of a distribution."""
    msg = f"Conditional entropy is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def mutual_information(distribution: SecondOrderDistributionLike, base: LogBase = None) -> ArrayLike:
    """Compute the mutual information of a distribution."""
    msg = f"Mutual information is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def max_probability_complement_of_expected(distribution: SecondOrderDistributionLike) -> ArrayLike:
    """Compute one minus the max probability of the expected value of a second-order distribution."""
    msg = f"Max probability complement of expected is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def expected_max_probability_complement(distribution: SecondOrderDistributionLike) -> ArrayLike:
    """Compute the expected value of one minus the max probability under a second-order distribution."""
    msg = f"Expected max probability complement is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@flexdispatch
def max_disagreement(distribution: SecondOrderDistributionLike) -> ArrayLike:
    """Compute the expected gap between each sample's max probability and its probability on the BMA argmax."""
    msg = f"Max disagreement is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)
