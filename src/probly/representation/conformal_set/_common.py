"""Common class representing credal sets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Self

from flextype import flexdispatch
from probly.representation.representation import Representation

if TYPE_CHECKING:
    from probly.representation.sample._common import Sample


type ConformalSetType = Literal["one_hot", "interval"]


class ConformalSet(Representation):
    """Base class for conformal sets."""

    type: ConformalSetType


class OneHotConformalSet[T](ConformalSet, ABC):
    """A conformal set represented as a one-hot vector."""

    type = "one_hot"

    @classmethod
    @abstractmethod
    def from_sample(cls, sample: Sample[T]) -> Self:
        """Create a one-hot conformal set from a sample.

        Args:
            sample: The sample to create the conformal set from.

        Returns:
            The created conformal set.
        """
        msg = "from_sample method not implemented."
        raise NotImplementedError(msg)


class IntervalConformalSet[T](ConformalSet, ABC):
    """A conformal set represented as an interval."""

    type = "interval"


def dispatch_on_sample(sample: Sample, **_kwargs: object) -> object:
    """Dispatch on the concrete sample object itself."""
    try:
        return sample.samples
    except Exception:  # noqa: BLE001
        return None


@flexdispatch
def create_onehot_conformal_set(sample: Sample) -> OneHotConformalSet:
    """Create a one-hot conformal set from a sample."""
    msg = "One-hot conformal set creation not implemented for this type."
    raise NotImplementedError(msg)


@flexdispatch
def create_interval_conformal_set(lower_bound: Sample, upper_bound: Sample) -> IntervalConformalSet:
    """Create an interval conformal set from lower and upper bound samples."""
    msg = "Interval conformal set creation not implemented for this type."
    raise NotImplementedError(msg)
