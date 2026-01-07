"""Common class representing credal sets."""

from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Literal, Self

if TYPE_CHECKING:
    from probly.representation.sampling.common_sample import Sample


type CredalSetType = Literal["categorical", "gaussian", "dirichlet"]


class CredalSet(ABC):
    """Base class for credal sets."""

    type: CredalSetType


class CategoricalCredalSet[T](CredalSet, metaclass=ABCMeta):
    """Base class for credal sets."""

    type = "categorical"

    @classmethod
    @abstractmethod
    def from_sample(cls, sample: Sample[T], distribution_axis: int = -1) -> Self:
        """Create a credal set from a finite sample.

        Args:
            sample: The sample to create the credal set from.
            distribution_axis: The axis containing the categorical probabilities.

        Returns:
            The created credal set.
        """
        msg = "from_sample method not implemented."
        raise NotImplementedError(msg)

    def lower(self) -> T:
        """Compute the lower envelope of the credal set."""
        msg = "lower method not implemented."
        raise NotImplementedError(msg)

    def upper(self) -> T:
        """Compute the upper envelope of the credal set."""
        msg = "upper method not implemented."
        raise NotImplementedError(msg)


class DiscreteCredalSet[T](CategoricalCredalSet[T]):
    """A credal set over a finite set of distributions."""


class ConvexCredalSet[T](CategoricalCredalSet[T]):
    """A credal set defined by the convex hull of a set of distributions."""


class DistanceBasedCredalSet[T](CategoricalCredalSet[T]):
    """A credal set defined by a distance metric around a central distribution."""


class ProbabilityIntervals[T](CategoricalCredalSet[T]):
    """A credal set defined by probability intervals over outcomes."""


class SingletonCredalSet[T](DiscreteCredalSet[T]):
    """A credal set containing a single distribution."""
