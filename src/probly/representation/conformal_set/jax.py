"""JAX-backed conformal sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self

if TYPE_CHECKING:
    from probly.representation.sample._common import Sample

from jax import numpy as jnp

from probly.representation._protected_axis.jax import JaxAxisProtected
from probly.representation.conformal_set._common import (
    IntervalConformalSet,
    OneHotConformalSet,
    create_interval_conformal_set,
    create_onehot_conformal_set,
)
from probly.representation.sample.jax import JaxArraySample


def _ensure_array_one_hot(value: object) -> jnp.ndarray:
    if isinstance(value, jnp.ndarray):
        if value.dtype == bool:
            return value
        if jnp.issubdtype(value.dtype, jnp.integer) and jnp.array_equal(value, value.astype(jnp.bool_)):
            return value.astype(bool)
    msg = "Value must be a one-hot encoded array of booleans or integers."
    raise ValueError(msg)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class JaxArrayOneHotConformalSet(JaxAxisProtected[Any], OneHotConformalSet):
    """One-hot conformal set backed by a JAX array."""

    array: jnp.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}

    def __post_init__(self) -> None:
        """Validate and coerce the array to a boolean one-hot array."""
        object.__setattr__(self, "array", _ensure_array_one_hot(self.array))

    @classmethod
    def from_array_sample(cls, sample: jnp.ndarray) -> Self:
        """Create a one-hot conformal set from a raw JAX array.

        Args:
            sample: A one-hot encoded boolean or integer array.

        Returns:
            The created conformal set.
        """
        if not isinstance(sample, jnp.ndarray):
            msg = "Expected jnp.ndarray for one-hot conformal sets."
            raise TypeError(msg)
        return cls(array=sample)

    @classmethod
    def from_sample(cls, sample: Sample[jnp.ndarray]) -> Self:
        """Create a one-hot conformal set from a sample.

        Args:
            sample: A sample containing a one-hot encoded array.

        Returns:
            The created conformal set.
        """
        array_sample = JaxArraySample.from_sample(sample)
        return cls.from_array_sample(array_sample.array)

    @property
    def set_size(self) -> jnp.ndarray:
        """Return the sizes of the conformal sets."""
        return jnp.sum(self.array, axis=-1)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class JaxArrayIntervalConformalSet(JaxAxisProtected[Any], IntervalConformalSet):
    """Interval conformal set backed by a JAX array storing lower and upper bounds."""

    array: jnp.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}

    @classmethod
    def from_array_samples(cls, lower: jnp.ndarray, upper: jnp.ndarray) -> Self:
        """Create an interval conformal set from lower and upper bound arrays.

        Args:
            lower: The lower bound array.
            upper: The upper bound array.

        Returns:
            The created interval conformal set.
        """
        if not isinstance(lower, jnp.ndarray) or not isinstance(upper, jnp.ndarray):
            msg = "Expected jnp.ndarray for interval conformal sets."
            raise TypeError(msg)
        return cls(array=jnp.stack([lower, upper], axis=-1))

    @classmethod
    def from_samples(cls, lower: JaxArraySample, upper: JaxArraySample) -> Self:
        """Create an interval conformal set from two JaxSamples.

        Args:
            lower: The lower bound sample.
            upper: The upper bound sample.

        Returns:
            The created interval conformal set.
        """
        if not isinstance(lower, JaxArraySample) or not isinstance(upper, JaxArraySample):
            msg = "Expected JaxArraySample for interval conformal sets."
            raise TypeError(msg)
        return cls.from_array_samples(lower.array, upper.array)

    @property
    def set_size(self) -> jnp.ndarray:
        """Return the sizes of the conformal sets."""
        return self.array[..., 1] - self.array[..., 0]


create_onehot_conformal_set.register(jnp.ndarray)(JaxArrayOneHotConformalSet.from_array_sample)
create_onehot_conformal_set.register(JaxArraySample)(JaxArrayOneHotConformalSet.from_sample)
create_interval_conformal_set.register(jnp.ndarray)(JaxArrayIntervalConformalSet.from_array_samples)
create_interval_conformal_set.register(JaxArraySample)(JaxArrayIntervalConformalSet.from_samples)
