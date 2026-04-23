"""Common class representing credal sets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Protocol, Self

from flextype import flexdispatch
from probly.representation.array_like import ArrayLike
from probly.representation.distribution import CategoricalDistribution
from probly.representation.representation import Representation

if TYPE_CHECKING:
    from probly.representation.sample._common import Sample


type CredalSetType = Literal["categorical", "gaussian", "dirichlet"]


class CredalSet(Representation):
    """Base class for credal sets."""

    type: CredalSetType


class CategoricalCredalSet[T: CategoricalDistribution](CredalSet, ABC):
    """Base class for credal sets."""

    type = "categorical"

    @classmethod
    @abstractmethod
    def from_sample(cls, sample: Sample[T]) -> Self:
        """Create a credal set from a finite sample.

        Args:
            sample: The sample to create the credal set from.

        Returns:
            The created credal set.
        """
        msg = "from_sample method not implemented."
        raise NotImplementedError(msg)

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Return the number of classes in the credal set."""
        msg = "num_classes property not implemented."
        raise NotImplementedError(msg)


class DiscreteCredalSet[T: CategoricalDistribution](CategoricalCredalSet[T]):
    """A credal set over a finite set of distributions."""


class ConvexCredalSet[T: CategoricalDistribution](CategoricalCredalSet[T]):
    """A credal set defined by the convex hull of a set of distributions."""


class DistanceBasedCredalSet[T: CategoricalDistribution](CategoricalCredalSet[T]):
    """A credal set defined by a distance metric around a central distribution."""


class ProbabilityIntervalsCredalSet[T: CategoricalDistribution](CategoricalCredalSet[T]):
    """A credal set defined by probability intervals over outcomes."""


class SingletonCredalSet[T: CategoricalDistribution](DiscreteCredalSet[T]):
    """A credal set containing a single distribution."""


class ProbabilityIntervalsFactory[S: Sample, C: ProbabilityIntervalsCredalSet](Protocol):
    """Factory protocol for probability-interval credal sets."""

    def __call__(self, sample: S) -> C:
        """Create a probability-interval credal set from a sample."""


@flexdispatch
def create_probability_intervals[T: CategoricalDistribution](sample: Sample[T]) -> ProbabilityIntervalsCredalSet:
    """Create a probability-interval credal set from a sample."""
    msg = f"No probability intervals factory registered for sample type {type(sample)}"
    raise NotImplementedError(msg)


@flexdispatch
def create_convex_credal_set[T: CategoricalDistribution](sample: Sample[T]) -> ConvexCredalSet[T]:
    """Create a convex credal set from a sample."""
    msg = f"No convex credal set factory registered for sample type {type(sample)}"
    raise NotImplementedError(msg)


@flexdispatch
def create_probability_intervals_from_lower_upper_array[T: ArrayLike](
    array: T,
) -> ProbabilityIntervalsCredalSet:
    """Create a probability-interval credal set from an array of lower and upper probabilities."""
    msg = f"No probability intervals factory registered for array type {type(array)}"
    raise NotImplementedError(msg)


@flexdispatch
def create_probability_intervals_from_bounds[T: ArrayLike](
    array: T,
    lower_bounds: T,
    upper_bounds: T,
) -> ProbabilityIntervalsCredalSet:
    """Create a probability-interval credal set from an array of predictions and lower and upper bounds."""
    msg = f"No probability intervals factory registered for array type {type(array)}"
    raise NotImplementedError(msg)
