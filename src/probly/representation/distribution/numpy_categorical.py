"""Numpy-based distribution representation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, override

import numpy as np
from scipy.special import logsumexp

from probly.representation._protected_axis.numpy import NumpyAxisProtected
from probly.representation.distribution._common import (
    CategoricalDistribution,
    CategoricalDistributionSample,
    create_categorical_distribution,
    create_categorical_distribution_from_logits,
)
from probly.representation.sample import NumpySample

if TYPE_CHECKING:
    from collections.abc import Callable


class NumpyCategoricalDistribution(CategoricalDistribution, NumpyAxisProtected[np.ndarray], ABC):
    """A categorical distribution stored as a numpy array.

    Shape: (..., num_classes)
    The last axis represents the category dimension.
    """

    @property
    @abstractmethod
    def array(self) -> np.ndarray:
        """Get the underlying array representing the categorical distribution."""

    @override
    def _postprocess_protected_values(self, values: dict[str, np.ndarray], func: Callable) -> dict[str, np.ndarray]:
        if func in (np.mean, np.average):
            values["array"] = self.probabilities

        return values

    @override
    def with_protected_values(
        self, values: dict[str, Any], func: Callable | None = None
    ) -> NumpyAxisProtected[np.ndarray]:
        """Return a copy with a replaced primary protected value."""
        if func in (np.mean, np.average) and not isinstance(self, NumpyProbabilityCategoricalDistribution):
            return NumpyProbabilityCategoricalDistribution(array=values["array"])

        return super().with_protected_values(values, func)

    @override
    @property
    def unnormalized_probabilities(self) -> np.ndarray:
        logits = self.logits
        return np.exp(logits - np.max(logits, axis=-1, keepdims=True))

    @override
    @property
    def probabilities(self) -> np.ndarray:
        unnormalized_probabilities = self.unnormalized_probabilities
        sums = np.sum(unnormalized_probabilities, axis=-1, keepdims=True)
        return unnormalized_probabilities / sums

    @override
    @property
    def logits(self) -> np.ndarray:
        return np.log(self.unnormalized_probabilities)

    @override
    @property
    def log_probabilities(self) -> np.ndarray:
        logits = self.logits
        return logits - logsumexp(logits, axis=-1, keepdims=True)

    @override
    @property
    def num_classes(self) -> int:
        """Get the number of classes."""
        return self.unnormalized_probabilities.shape[-1]

    @override
    def sample(
        self,
        num_samples: int = 1,
        rng: np.random.Generator | None = None,
    ) -> NumpySample[np.ndarray]:
        """Sample from the categorical distribution (NumPy backend)."""
        if rng is None:
            rng = np.random.default_rng()

        flat_probabilities = self.probabilities.reshape((-1, self.num_classes))
        flat_samples = np.empty((num_samples, flat_probabilities.shape[0]), dtype=np.int64)

        for i, probabilities in enumerate(flat_probabilities):
            flat_samples[:, i] = rng.choice(a=self.num_classes, size=num_samples, p=probabilities)

        samples = flat_samples.reshape((num_samples, *self.shape))
        return NumpySample(array=samples, sample_axis=0)


@create_categorical_distribution.register(np.ndarray)
@dataclass(frozen=True, slots=True, weakref_slot=True)
class NumpyProbabilityCategoricalDistribution(NumpyCategoricalDistribution):
    """A categorical distribution represented by unnormalized probabilities."""

    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}
    permitted_functions: ClassVar[set[Callable]] = {np.mean, np.average}

    def __post_init__(self) -> None:
        """Validate the concentration parameters."""
        if not isinstance(self.array, np.ndarray):
            msg = "probabilities must be a numpy ndarray."
            raise TypeError(msg)

        if self.array.ndim < 1:
            msg = "probabilities must have at least one dimension."
            raise ValueError(msg)
        if np.any(self.array < 0):
            msg = "Relative probabilities must be non-negative."
            raise ValueError(msg)

    @override
    @property
    def unnormalized_probabilities(self) -> np.ndarray:
        return self.array

    @override
    def __eq__(self, value: Any) -> np.ndarray:  # ty: ignore[invalid-method-override]  # noqa: PYI032
        """Vectorized equality comparison."""
        if isinstance(value, NumpyCategoricalDistribution):
            eq = np.equal(self.probabilities, value.probabilities)
        else:
            eq = np.equal(self.array, value)
        return np.all(eq, axis=-1)

    def __hash__(self) -> int:
        """Return an identity-based hash.

        We intentionally bypass ``super()`` here because protocol-heavy MROs can
        produce invalid ``super(type, obj)`` bindings at runtime. ``object``'s
        hash gives per-instance identity semantics.
        """
        return object.__hash__(self)


@create_categorical_distribution_from_logits.register(np.ndarray)
@dataclass(frozen=True, slots=True, weakref_slot=True)
class NumpyLogitCategoricalDistribution(NumpyCategoricalDistribution):
    """A categorical distribution represented by logits."""

    array: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"array": 1}
    permitted_functions: ClassVar[set[Callable]] = {np.mean, np.average}

    def __post_init__(self) -> None:
        """Validate the concentration parameters."""
        if not isinstance(self.array, np.ndarray):
            msg = "logits must be a numpy ndarray."
            raise TypeError(msg)

        if self.array.ndim < 1:
            msg = "logits must have at least one dimension."
            raise ValueError(msg)

    @override
    @property
    def logits(self) -> np.ndarray:
        return self.array

    @override
    def __eq__(self, value: Any) -> np.ndarray:  # ty: ignore[invalid-method-override]  # noqa: PYI032
        """Vectorized equality comparison."""
        if isinstance(value, NumpyCategoricalDistribution):
            eq = np.equal(self.log_probabilities, value.log_probabilities)
        else:
            eq = np.equal(self.array, value)
        return np.all(eq, axis=-1)

    def __hash__(self) -> int:
        """Return an identity-based hash.

        We intentionally bypass ``super()`` here because protocol-heavy MROs can
        produce invalid ``super(type, obj)`` bindings at runtime. ``object``'s
        hash gives per-instance identity semantics.
        """
        return object.__hash__(self)


class NumpyCategoricalDistributionSample(  # ty:ignore[conflicting-metaclass]
    CategoricalDistributionSample[NumpyCategoricalDistribution],
    NumpySample[NumpyCategoricalDistribution],
):
    """Sample type for empirical second-order categorical distributions."""

    sample_space: ClassVar[type[CategoricalDistribution]] = NumpyCategoricalDistribution

    @override
    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        return super().__instancehook__(instance)


@create_categorical_distribution.register((list, tuple))
def _create_numpy_categorical_distribution_from_sequence(
    data: list[Any] | tuple[Any, ...],
) -> NumpyCategoricalDistribution:
    return NumpyProbabilityCategoricalDistribution(np.asarray(data))
