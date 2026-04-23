"""Numpy-based distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, override

import numpy as np

from probly.representation._protected_axis.array import ArrayAxisProtected
from probly.representation.distribution._common import (
    CategoricalDistribution,
    CategoricalDistributionSample,
    create_categorical_distribution,
)
from probly.representation.sample import ArraySample

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@create_categorical_distribution.register(np.ndarray)
@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArrayCategoricalDistribution(
    ArrayAxisProtected[np.ndarray],
    CategoricalDistribution,
):
    """A categorical distribution stored as a numpy array.

    Shape: (..., num_classes)
    The last axis represents the category dimension.
    """

    unnormalized_probabilities: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"unnormalized_probabilities": 1}
    permitted_functions: ClassVar[set[Callable]] = {np.mean}

    def __post_init__(self) -> None:
        """Validate the concentration parameters."""
        if not isinstance(self.unnormalized_probabilities, np.ndarray):
            msg = "probabilities must be a numpy ndarray."
            raise TypeError(msg)

        if self.unnormalized_probabilities.ndim < 1:
            msg = "probabilities must have at least one dimension."
            raise ValueError(msg)

        if self._is_bernoulli:
            if np.any(self.unnormalized_probabilities < 0) or np.any(self.unnormalized_probabilities > 1):
                msg = "Bernoulli probabilities must be in the range [0, 1]."
                raise ValueError(msg)
        elif np.any(self.unnormalized_probabilities < 0):
            msg = "Relative probabilities must be non-negative."
            raise ValueError(msg)

    @property
    def _is_bernoulli(self) -> bool:
        return self.unnormalized_probabilities.shape[-1] == 1

    def _bernoulli_probability(self) -> np.ndarray:
        return self.unnormalized_probabilities[..., 0]

    @override
    @property
    def probabilities(self) -> np.ndarray:
        if self._is_bernoulli:
            p = self._bernoulli_probability()
            q = 1 - p
            return np.stack((p, q), axis=-1)

        sums = np.sum(self.unnormalized_probabilities, axis=-1, keepdims=True)

        return self.unnormalized_probabilities / sums

    @override
    @property
    def num_classes(self) -> int:
        """Get the number of classes."""
        if self._is_bernoulli:
            return 2
        return self.unnormalized_probabilities.shape[-1]

    @override
    def sample(
        self,
        num_samples: int = 1,
        rng: np.random.Generator | None = None,
    ) -> ArraySample[np.ndarray]:
        """Sample from the categorical distribution (NumPy backend)."""
        if rng is None:
            rng = np.random.default_rng()

        if self._is_bernoulli:
            probabilities = self._bernoulli_probability()
            samples = rng.binomial(1, probabilities, size=(num_samples, *self.shape))
            return ArraySample(array=samples, sample_axis=0)

        flat_probabilities = self.probabilities.reshape((-1, self.num_classes))
        flat_samples = np.empty((num_samples, flat_probabilities.shape[0]), dtype=np.int64)

        for i, probabilities in enumerate(flat_probabilities):
            flat_samples[:, i] = rng.choice(a=self.num_classes, size=num_samples, p=probabilities)

        samples = flat_samples.reshape((num_samples, *self.shape))
        return ArraySample(array=samples, sample_axis=0)

    @override
    def __iter__(self) -> Iterator[Any]:
        return self.unnormalized_probabilities.__iter__()

    @override
    def __eq__(self, value: Any) -> np.ndarray:  # ty: ignore[invalid-method-override]  # noqa: PYI032
        """Vectorized equality comparison."""
        if isinstance(value, ArrayCategoricalDistribution):
            eq = np.equal(self.probabilities, value.probabilities)
        else:
            eq = np.equal(self.unnormalized_probabilities, value)
        return np.all(eq, axis=-1)

    def __hash__(self) -> int:
        """Return an identity-based hash.

        We intentionally bypass ``super()`` here because protocol-heavy MROs can
        produce invalid ``super(type, obj)`` bindings at runtime. ``object``'s
        hash gives per-instance identity semantics.
        """
        return object.__hash__(self)


class ArrayCategoricalDistributionSample(  # ty:ignore[conflicting-metaclass]
    CategoricalDistributionSample[ArrayCategoricalDistribution],
    ArraySample[ArrayCategoricalDistribution],
):
    """Sample type for empirical second-order categorical distributions."""

    sample_space: ClassVar[type[CategoricalDistribution]] = ArrayCategoricalDistribution

    @override
    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        return super().__instancehook__(instance)


@create_categorical_distribution.register((list, tuple))
def _create_array_categorical_distribution_from_sequence(
    data: list[Any] | tuple[Any, ...],
) -> ArrayCategoricalDistribution:
    return ArrayCategoricalDistribution(np.asarray(data))


@create_categorical_distribution.register(ArrayCategoricalDistribution)
def _create_array_categorical_distribution_from_instance(
    data: ArrayCategoricalDistribution,
) -> ArrayCategoricalDistribution:
    return data
