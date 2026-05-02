"""Common deciders for categorical distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flextype import flexdispatch
from probly.representation.credal_set._common import CategoricalCredalSet
from probly.representation.distribution import CategoricalDistribution, CategoricalDistributionSample
from probly.representation.distribution._common import DirichletDistribution

if TYPE_CHECKING:
    from probly.representation.representation import Representation


@flexdispatch
def categorical_from_mean(representation: Representation) -> CategoricalDistribution:
    """Create a categorical distribution from the mean of a representation."""
    msg = f"categorical_from_mean decider not supported for {type(representation).__name__}."
    raise NotImplementedError(msg)


@flexdispatch
def categorical_from_maximin(representation: Representation) -> CategoricalDistribution:
    """Create a one-hot categorical distribution via the maximin decision rule.

    For each batch element, the maximin rule selects the class whose lower probability is highest,
    returning a degenerate categorical distribution that puts mass one on that class.
    """
    msg = f"categorical_from_maximin decider not supported for {type(representation).__name__}."
    raise NotImplementedError(msg)


@categorical_from_mean.register(CategoricalDistribution)
def _(distribution: CategoricalDistribution) -> CategoricalDistribution:
    return distribution


@categorical_from_mean.register(CategoricalDistributionSample)
def _(sample: CategoricalDistributionSample) -> CategoricalDistribution:
    return sample.sample_mean()


@categorical_from_mean.register(DirichletDistribution)
def _(distribution: DirichletDistribution) -> CategoricalDistribution:
    return distribution.mean


@categorical_from_mean.register(CategoricalCredalSet)
def _(credal_set: CategoricalCredalSet) -> CategoricalDistribution:
    return credal_set.barycenter
