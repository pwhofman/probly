"""Common class representing credal sets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Protocol, Self

from lazy_dispatch import lazydispatch
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

    def __call__(self, sample: S, distribution_axis: int = -1) -> C:
        """Create a probability-interval credal set from a sample."""


def dispatch_on_sample(sample: Sample, **_kwargs: object) -> object:
    """Dispatch on the concrete sample object itself."""
    try:
        return sample.samples
    except Exception:  # noqa: BLE001
        return None


@lazydispatch(dispatch_on=dispatch_on_sample)
def create_probability_intervals(sample: Sample) -> ProbabilityIntervalsCredalSet:
    """Create a probability-interval credal set from a sample."""
    msg = f"No probability intervals factory registered for sample type {type(sample)}"
    raise NotImplementedError(msg)


@lazydispatch
def create_convex_credal_set[T: CategoricalDistribution](
    data: Sample[T], distribution_axis: int = -1
) -> ConvexCredalSet[T]:
    """Create a convex credal set from a sample."""
    msg = f"No convex credal set factory registered for data type {type(data)}"
    raise NotImplementedError(msg)
