"""NumPy-backed categorical credal set representations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Self, override

import numpy as np

from probly.representation._protected_axis.array import ArrayAxisProtected
from probly.representation.credal_set._common import (
    CategoricalCredalSet,
    ConvexCredalSet,
    DiscreteCredalSet,
    DistanceBasedCredalSet,
    ProbabilityIntervalsCredalSet,
    SingletonCredalSet,
    create_convex_credal_set,
    create_probability_intervals,
)
from probly.representation.distribution import ArrayCategoricalDistribution
from probly.representation.sample import ArraySample

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from probly.representation.sample._common import Sample


def _ensure_array_categorical_distribution(value: object) -> ArrayCategoricalDistribution:
    if isinstance(value, ArrayCategoricalDistribution):
        return value
    return ArrayCategoricalDistribution(np.asarray(value))


def _sample_probabilities(sample: ArraySample[ArrayCategoricalDistribution]) -> np.ndarray:
    sample_values = sample.samples
    if not isinstance(sample_values, ArrayCategoricalDistribution):
        msg = "Array categorical credal sets require samples of ArrayCategoricalDistribution."
        raise TypeError(msg)

    return sample_values.unnormalized_probabilities


class ArrayCategoricalCredalSet(CategoricalCredalSet, ABC):
    """Base class for NumPy-backed categorical credal sets."""

    @classmethod
    def from_sample(cls, sample: Sample[ArrayCategoricalDistribution]) -> Self:
        """Create a credal set from a sample of categorical distributions."""
        array_sample = ArraySample.from_sample(sample)
        if not isinstance(array_sample.array, ArrayCategoricalDistribution):
            msg = "Expected ArraySample[ArrayCategoricalDistribution] for categorical credal sets."
            raise TypeError(msg)
        return cls.from_array_sample(array_sample)

    @classmethod
    @abstractmethod
    def from_array_sample(
        cls,
        sample: ArraySample[ArrayCategoricalDistribution],
    ) -> Self:
        """Create a credal set from categorical distribution samples."""

    @abstractmethod
    def lower(self) -> np.ndarray:
        """Return the lower probabilities of the credal set."""

    @abstractmethod
    def upper(self) -> np.ndarray:
        """Return the upper probabilities of the credal set."""


@dataclass(frozen=True, slots=True, weakref_slot=True)  # ty:ignore[conflicting-metaclass]
class ArrayDiscreteCredalSet(
    ArrayAxisProtected[ArrayCategoricalDistribution],
    ArrayCategoricalCredalSet,
    DiscreteCredalSet,
):
    """A finite set of categorical distributions."""

    array: ArrayCategoricalDistribution
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}

    def __post_init__(self) -> None:
        """Validate that the array contains valid categorical distributions."""
        object.__setattr__(self, "array", _ensure_array_categorical_distribution(self.array))

    @override
    @classmethod
    def from_array_sample(cls, sample: ArraySample[ArrayCategoricalDistribution]) -> Self:
        probabilities = _sample_probabilities(sample)
        members = np.moveaxis(probabilities, 0, -2)
        return cls(array=ArrayCategoricalDistribution(members))

    @override
    @property
    def num_classes(self) -> int:
        """Return the number of classes in the credal set."""
        return self.array.num_classes

    @override
    def lower(self) -> np.ndarray:
        """Return the lower probabilities of the credal set."""
        return np.min(self.array.unnormalized_probabilities, axis=0)

    @override
    def upper(self) -> np.ndarray:
        """Return the upper probabilities of the credal set."""
        return np.max(self.array.unnormalized_probabilities, axis=0)


@dataclass(frozen=True, slots=True, weakref_slot=True)  # ty:ignore[conflicting-metaclass]
class ArrayConvexCredalSet(
    ArrayAxisProtected[ArrayCategoricalDistribution],
    ArrayCategoricalCredalSet,
    ConvexCredalSet,
):
    """A convex hull over a finite set of categorical distributions."""

    array: ArrayCategoricalDistribution
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}

    def __post_init__(self) -> None:
        """Validate that the array contains valid categorical distributions."""
        object.__setattr__(self, "array", _ensure_array_categorical_distribution(self.array))

    @override
    @classmethod
    def from_array_sample(
        cls,
        sample: ArraySample[ArrayCategoricalDistribution],
    ) -> Self:
        probabilities = _sample_probabilities(sample)
        vertices = np.moveaxis(probabilities, 0, -2)
        return cls(array=ArrayCategoricalDistribution(vertices))

    @override
    @property
    def num_classes(self) -> int:
        """Return the number of classes in the credal set."""
        return self.array.num_classes

    @override
    def lower(self) -> np.ndarray:
        """Return the lower probabilities of the credal set."""
        return np.min(self.array.unnormalized_probabilities, axis=0)

    @override
    def upper(self) -> np.ndarray:
        """Return the upper probabilities of the credal set."""
        return np.max(self.array.unnormalized_probabilities, axis=0)


@dataclass(frozen=True, slots=True, weakref_slot=True)  # ty:ignore[conflicting-metaclass]
class ArrayDistanceBasedCredalSet(
    ArrayAxisProtected[ArrayCategoricalDistribution],
    ArrayCategoricalCredalSet,
    DistanceBasedCredalSet,
):
    """Distance-based credal set around a nominal categorical distribution."""

    nominal: ArrayCategoricalDistribution
    radius: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"nominal": 0, "radius": 0}

    def __post_init__(self) -> None:
        """Validate that nominal is a valid categorical distribution and radius is non-negative."""
        object.__setattr__(self, "nominal", _ensure_array_categorical_distribution(self.nominal))
        object.__setattr__(self, "radius", np.asarray(self.radius))

    @override
    @classmethod
    def from_array_sample(cls, sample: ArraySample[ArrayCategoricalDistribution]) -> Self:
        probabilities = _sample_probabilities(sample)
        nominal = np.mean(probabilities, axis=0)
        diff = np.abs(probabilities - nominal)
        tv_dists = 0.5 * np.sum(diff, axis=-1)
        radius = np.max(tv_dists, axis=0)
        return cls(
            nominal=ArrayCategoricalDistribution(nominal),
            radius=np.asarray(radius),
        )

    @override
    def __array__(self, dtype: DTypeLike | None = None, copy: bool | None = None) -> np.ndarray:
        return np.asarray(self.nominal.unnormalized_probabilities, dtype=dtype, copy=copy)

    @override
    @property
    def num_classes(self) -> int:
        """Return the number of classes in the credal set."""
        return self.nominal.num_classes

    @override
    def lower(self) -> np.ndarray:
        """Compute the lower envelope of the credal set.

        For L1/TV distance, the tightest element-wise lower bound is max(0, nominal - radius).
        """
        # Ensure radius is broadcastable to nominal (add last dim if needed)
        nominal = self.nominal.unnormalized_probabilities
        r = self.radius
        if isinstance(r, np.ndarray) and r.ndim == nominal.ndim - 1:
            r = np.expand_dims(r, axis=-1)

        return np.clip(nominal - r, 0.0, 1.0)

    @override
    def upper(self) -> np.ndarray:
        """Compute the upper envelope of the credal set.

        For L1/TV distance, the tightest element-wise upper bound is min(1, nominal + radius).
        """
        # Ensure radius is broadcastable to nominal (add last dim if needed)
        nominal = self.nominal.unnormalized_probabilities
        r = self.radius
        if isinstance(r, np.ndarray) and r.ndim == nominal.ndim - 1:
            r = np.expand_dims(r, axis=-1)

        return np.clip(nominal + r, 0.0, 1.0)


@dataclass(frozen=True, slots=True, weakref_slot=True)  # ty:ignore[conflicting-metaclass]
class ArrayProbabilityIntervalsCredalSet(
    ArrayAxisProtected[ArrayCategoricalDistribution],
    ArrayCategoricalCredalSet,
    ProbabilityIntervalsCredalSet,
):
    """Credal set represented by lower/upper categorical bounds."""

    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"lower_bounds": 1, "upper_bounds": 1}

    def __post_init__(self) -> None:
        """Validate that lower and upper bounds have the same shape and are valid distributions."""
        if self.lower_bounds.shape != self.upper_bounds.shape:
            msg = "Lower and upper bounds must have the same shape."
            raise ValueError(msg)

    @override
    @classmethod
    def from_array_sample(cls, sample: ArraySample[ArrayCategoricalDistribution]) -> Self:
        probabilities = _sample_probabilities(sample)
        lower_bounds = np.min(probabilities, axis=0)
        upper_bounds = np.max(probabilities, axis=0)
        return cls(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

    @override
    def __array__(self, dtype: DTypeLike | None = None, copy: bool | None = None) -> np.ndarray:
        stacked = np.stack([self.lower_bounds, self.upper_bounds], axis=-2)
        return np.asarray(stacked, dtype=dtype, copy=copy)

    def width(self) -> np.ndarray:
        """Compute interval width for each class."""
        return self.upper_bounds - self.lower_bounds

    def contains(self, probabilities: np.ndarray) -> np.ndarray:
        """Check whether probabilities are inside the intervals."""
        within_bounds = (probabilities >= self.lower_bounds) & (probabilities <= self.upper_bounds)
        return np.all(within_bounds, axis=-1)

    @override
    @property
    def num_classes(self) -> int:
        """Return the number of classes in the credal set."""
        return self.lower_bounds.shape[-1]

    @override
    def lower(self) -> np.ndarray:
        """Return the lower probabilities of the credal set."""
        return self.lower_bounds

    @override
    def upper(self) -> np.ndarray:
        """Return the upper probabilities of the credal set."""
        return self.upper_bounds


@dataclass(frozen=True, slots=True, weakref_slot=True)  # ty:ignore[conflicting-metaclass]
class ArraySingletonCredalSet(
    ArrayAxisProtected[ArrayCategoricalDistribution],
    ArrayCategoricalCredalSet,
    SingletonCredalSet,
):
    """A singleton credal set with one precise categorical distribution."""

    array: ArrayCategoricalDistribution
    protected_axes: ClassVar[dict[str, int]] = {"array": 0}

    def __post_init__(self) -> None:
        """Validate that the array contains a valid categorical distribution."""
        object.__setattr__(self, "array", _ensure_array_categorical_distribution(self.array))

    @override
    @classmethod
    def from_array_sample(cls, sample: ArraySample[ArrayCategoricalDistribution]) -> Self:
        probabilities = _sample_probabilities(sample)
        return cls(array=ArrayCategoricalDistribution(np.mean(probabilities, axis=0)))

    @override
    @property
    def num_classes(self) -> int:
        """Return the number of classes in the credal set."""
        return self.array.num_classes

    @override
    def lower(self) -> np.ndarray:
        """Return the lower probabilities of the credal set."""
        return self.array.unnormalized_probabilities

    @override
    def upper(self) -> np.ndarray:
        """Return the upper probabilities of the credal set."""
        return self.array.unnormalized_probabilities


create_probability_intervals.register(ArrayCategoricalDistribution, ArrayProbabilityIntervalsCredalSet.from_sample)
create_convex_credal_set.register(ArraySample, ArrayConvexCredalSet.from_array_sample)
