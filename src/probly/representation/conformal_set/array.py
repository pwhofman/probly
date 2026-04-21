"""NumPy-backed conformal sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Self

if TYPE_CHECKING:
    from probly.representation.sample._common import Sample

import numpy as np

from probly.representation._protected_axis.array import ArrayAxisProtected
from probly.representation.conformal_set._common import (
    IntervalConformalSet,
    OneHotConformalSet,
    create_interval_conformal_set,
    create_onehot_conformal_set,
)
from probly.representation.sample.array import ArraySample


def _ensure_array_one_hot(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.dtype == bool:
            return value
        if value.dtype == int and np.array_equal(value, value.astype(np.bool_)):  # ty: ignore[no-matching-overload]
            return value.astype(bool)  # ty: ignore[no-matching-overload]
    msg = "Value must be a one-hot encoded array of booleans or integers."
    raise ValueError(msg)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArrayOneHotConformalSet(ArrayAxisProtected[ArraySample], OneHotConformalSet):
    """One-hot conformal set backed by a NumPy array."""

    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}

    def __post_init__(self) -> None:
        """Validate and coerce the array to a boolean one-hot array."""
        object.__setattr__(self, "array", _ensure_array_one_hot(self.array))

    @classmethod
    def from_array_sample(cls, sample: np.ndarray) -> Self:
        """Create a one-hot conformal set from a raw NumPy array.

        Args:
            sample: A one-hot encoded boolean or integer array.

        Returns:
            The created conformal set.
        """
        if not isinstance(sample, np.ndarray):
            msg = "Expected np.ndarray for one-hot conformal sets."
            raise TypeError(msg)
        return cls(array=sample)

    @classmethod
    def from_sample(cls, sample: Sample[np.ndarray]) -> Self:
        """Create a one-hot conformal set from a sample.

        Args:
            sample: A sample containing a one-hot encoded array.

        Returns:
            The created conformal set.
        """
        array_sample = ArraySample.from_sample(sample)
        return cls.from_array_sample(array_sample.array)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArrayIntervalConformalSet(ArrayAxisProtected[ArraySample], IntervalConformalSet):
    """Interval conformal set backed by a NumPy array storing lower and upper bounds."""

    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}

    @classmethod
    def from_array_samples(cls, lower: np.ndarray, upper: np.ndarray) -> Self:
        """Create an interval conformal set from lower and upper bound arrays.

        Args:
            lower: The lower bound array.
            upper: The upper bound array.

        Returns:
            The created interval conformal set.
        """
        if not isinstance(lower, np.ndarray) or not isinstance(upper, np.ndarray):
            msg = "Expected np.ndarray for interval conformal sets."
            raise TypeError(msg)
        return cls(array=np.stack([lower, upper], axis=-1))

    @classmethod
    def from_samples(cls, lower: ArraySample, upper: ArraySample) -> Self:
        """Create an interval conformal set from two ArraySamples.

        Args:
            lower: The lower bound sample.
            upper: The upper bound sample.

        Returns:
            The created interval conformal set.
        """
        if not isinstance(lower, ArraySample) or not isinstance(upper, ArraySample):
            msg = "Expected ArraySample for interval conformal sets."
            raise TypeError(msg)
        return cls.from_array_samples(lower.array, upper.array)


create_onehot_conformal_set.register(np.ndarray)(ArrayOneHotConformalSet.from_array_sample)
create_onehot_conformal_set.register(ArraySample)(ArrayOneHotConformalSet.from_sample)
create_interval_conformal_set.register(np.ndarray)(ArrayIntervalConformalSet.from_array_samples)
create_interval_conformal_set.register(ArraySample)(ArrayIntervalConformalSet.from_samples)
