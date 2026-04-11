"""Common definitions of distribution measures."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lazy_dispatch import lazydispatch
from probly.representation.sample._common import Sample

if TYPE_CHECKING:
    from probly.representation.array_like import ArrayLike
    from probly.representation.distribution import Distribution, SecondOrderDistribution


@lazydispatch
def entropy(distribution: Distribution) -> ArrayLike:
    """Compute the entropy of a distribution."""
    msg = f"Entropy is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


type SecondOrderDistributionLike = SecondOrderDistribution | Sample[Distribution]


@lazydispatch
def entropy_of_expected_value(distribution: SecondOrderDistributionLike) -> ArrayLike:
    """Compute the entropy of the expected value of a second-order distribution."""
    msg = f"Entropy of expected value is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@lazydispatch
def conditional_entropy(distribution: SecondOrderDistributionLike) -> ArrayLike:
    """Compute the conditional entropy of a distribution."""
    msg = f"Conditional entropy is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)


@lazydispatch
def mutual_information(distribution: SecondOrderDistributionLike) -> ArrayLike:
    """Compute the mutual information of a distribution."""
    msg = f"Mutual information is not supported for distributions of type {type(distribution)}."
    raise NotImplementedError(msg)
