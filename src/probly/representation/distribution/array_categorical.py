"""Numpy-based distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, override

import numpy as np

from probly.representation._protected_axis.array import ArrayAxisProtected
from probly.representation.array_like import NumpyArrayLike
from probly.representation.distribution._common import CategoricalDistribution, create_categorical_distribution
from probly.representation.sample import ArraySample

if TYPE_CHECKING:
    from collections.abc import Iterator


@create_categorical_distribution.register(np.ndarray)
@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArrayCategoricalDistribution(
    ArrayAxisProtected,
    NumpyArrayLike[Any],
    CategoricalDistribution,
):
    """A categorical distribution stored as a numpy array.

    Shape: (..., num_classes)
    The last axis represents the category dimension.
    """

    probabilities: np.ndarray
    protected_axes: ClassVar[int] = 1

    @property
    def _is_bernoulli(self) -> bool:
        return self.probabilities.shape[-1] == 1

    def _bernoulli_probability(self) -> np.ndarray:
        return self.probabilities[..., 0]

    def _normalized_probabilities(self) -> np.ndarray:
        if self._is_bernoulli:
            msg = "Bernoulli distributions do not use categorical normalization."
            raise ValueError(msg)

        sums = np.sum(self.probabilities, axis=-1, keepdims=True)
        if np.any(sums <= 0):
            msg = "Relative probabilities must have strictly positive sum along the last axis."
            raise ValueError(msg)

        return self.probabilities / sums

    def __post_init__(self) -> None:
        """Validate the concentration parameters."""
        if not isinstance(self.probabilities, np.ndarray):
            msg = "probabilities must be a numpy ndarray."
            raise TypeError(msg)

        if self.probabilities.ndim < 1:
            msg = "probabilities must have at least one dimension."
            raise ValueError(msg)

        if self._is_bernoulli:
            if np.any(self.probabilities < 0) or np.any(self.probabilities > 1):
                msg = "Bernoulli probabilities must be in the range [0, 1]."
                raise ValueError(msg)
        elif np.any(self.probabilities < 0):
            msg = "Relative probabilities must be non-negative."
            raise ValueError(msg)

    @override
    def with_protected_array(self, array: np.ndarray) -> Self:
        return type(self)(array)

    @override
    @property
    def num_classes(self) -> int:
        """Get the number of classes."""
        if self._is_bernoulli:
            return 2
        return self.probabilities.shape[-1]

    @override
    @property
    def entropy(self) -> float:
        """Compute the entropy of the categorical distribution."""
        if self._is_bernoulli:
            p = self._bernoulli_probability()
            q = 1 - p
            log_p = np.where(p > 0, np.log(p), 0.0)
            log_q = np.where(q > 0, np.log(q), 0.0)
            return -(p * log_p + q * log_q)

        p = self._normalized_probabilities()
        log_p = np.where(p > 0, np.log(p), 0.0)
        return -np.sum(p * log_p, axis=-1)

    @override
    def sample(
        self,
        num_samples: int = 1,
        rng: np.random.Generator | None = None,
    ) -> ArraySample:
        """Sample from the categorical distribution (NumPy backend)."""
        if rng is None:
            rng = np.random.default_rng()

        if self._is_bernoulli:
            probabilities = self._bernoulli_probability()
            samples = rng.binomial(1, probabilities, size=(num_samples, *self.shape))
            return ArraySample(array=cast("NumpyArrayLike[Any]", samples), sample_axis=0)

        flat_probabilities = self._normalized_probabilities().reshape((-1, self.num_classes))
        flat_samples = np.empty((num_samples, flat_probabilities.shape[0]), dtype=np.int64)

        for i, probabilities in enumerate(flat_probabilities):
            flat_samples[:, i] = rng.choice(a=self.num_classes, size=num_samples, p=probabilities)

        samples = flat_samples.reshape((num_samples, *self.shape))
        return ArraySample(array=samples, sample_axis=0)  # ty:ignore[invalid-argument-type]

    @override
    def __iter__(self) -> Iterator[Any]:
        return self.probabilities.__iter__()

    def __eq__(self, value: Any) -> Self:  # ty: ignore[invalid-method-override]  # noqa: ANN401, PYI032
        """Vectorized equality comparison."""
        if isinstance(value, ArrayCategoricalDistribution):
            return np.equal(self.probabilities, value.probabilities)  # ty: ignore[invalid-return-type]
        return np.equal(self.probabilities, value)

    def __hash__(self) -> int:
        """Compute the hash of the distribution."""
        return super().__hash__()


@create_categorical_distribution.register((list, tuple))
def _create_array_categorical_distribution_from_sequence(
    data: list[Any] | tuple[Any, ...],
) -> ArrayCategoricalDistribution:
    return ArrayCategoricalDistribution(probabilities=np.asarray(data))


@create_categorical_distribution.register(ArrayCategoricalDistribution)
def _create_array_categorical_distribution_from_instance(
    data: ArrayCategoricalDistribution,
) -> ArrayCategoricalDistribution:
    return data
