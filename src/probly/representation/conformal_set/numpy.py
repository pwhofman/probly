"""NumPy-backed conformal sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Self

if TYPE_CHECKING:
    from probly.representation.sample._common import Sample

import numpy as np

from probly.representation._protected_axis.numpy import NumpyAxisProtected
from probly.representation.conformal_set._common import (
    IntervalConformalSet,
    OneHotConformalSet,
    create_interval_conformal_set,
    create_onehot_conformal_set,
)
from probly.representation.sample.numpy import NumpySample


def _ensure_numpy_one_hot(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.dtype == bool:
            return value
        if value.dtype == int and np.array_equal(value, value.astype(np.bool_)):  # ty: ignore[no-matching-overload]
            return value.astype(bool)  # ty: ignore[no-matching-overload]
    msg = "Value must be a one-hot encoded array of booleans or integers."
    raise ValueError(msg)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class NumpyOneHotConformalSet(NumpyAxisProtected[NumpySample], OneHotConformalSet):
    """One-hot conformal set backed by a NumPy array."""

    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}

    def __post_init__(self) -> None:
        """Validate and coerce the array to a boolean one-hot array."""
        object.__setattr__(self, "array", _ensure_numpy_one_hot(self.array))

    @classmethod
    def from_numpy_sample(cls, sample: np.ndarray) -> Self:
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
        array_sample = NumpySample.from_sample(sample)
        return cls.from_numpy_sample(array_sample.array)

    @property
    def set_size(self) -> np.ndarray:
        """Return the sizes of the conformal sets."""
        return np.sum(self.array, axis=-1)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class NumpyIntervalConformalSet(NumpyAxisProtected[NumpySample], IntervalConformalSet):
    """Interval conformal set backed by a NumPy array storing lower and upper bounds."""

    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}

    @classmethod
    def from_numpy_samples(cls, lower: np.ndarray, upper: np.ndarray) -> Self:
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
    def from_samples(cls, lower: NumpySample, upper: NumpySample) -> Self:
        """Create an interval conformal set from two NumpySamples.

        Args:
            lower: The lower bound sample.
            upper: The upper bound sample.

        Returns:
            The created interval conformal set.
        """
        if not isinstance(lower, NumpySample) or not isinstance(upper, NumpySample):
            msg = "Expected NumpySample for interval conformal sets."
            raise TypeError(msg)
        return cls.from_numpy_samples(lower.array, upper.array)

    @property
    def set_size(self) -> np.ndarray:
        """Return the sizes of the conformal sets."""
        return self.array[..., 1] - self.array[..., 0]


create_onehot_conformal_set.register(np.ndarray)(NumpyOneHotConformalSet.from_numpy_sample)
create_onehot_conformal_set.register(NumpySample)(NumpyOneHotConformalSet.from_sample)
create_interval_conformal_set.register(np.ndarray)(NumpyIntervalConformalSet.from_numpy_samples)
create_interval_conformal_set.register(NumpySample)(NumpyIntervalConformalSet.from_samples)
