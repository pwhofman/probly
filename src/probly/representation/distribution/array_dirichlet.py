"""Numpy-based distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self, override

import numpy as np

from probly.representation._protected_axis.array import ArrayAxisProtected
from probly.representation.distribution._common import DirichletDistribution
from probly.representation.distribution.array_categorical import ArrayCategoricalDistribution
from probly.representation.sample import ArraySample

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import DTypeLike


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArrayDirichletDistribution(
    ArrayAxisProtected[np.ndarray],
    DirichletDistribution[ArrayCategoricalDistribution],
):
    """A Dirichlet distribution stored as a numpy array.

    Shape: (..., num_classes)
    The last axis represents the category dimension.
    """

    alphas: np.ndarray
    protected_axes: ClassVar[dict[str, int]] = {"alphas": 1}
    permitted_ufuncs: ClassVar[dict[np.ufunc, list[str]]] = {
        np.add: ["__call__"],
        np.subtract: ["__call__"],
    }

    def __post_init__(self) -> None:
        """Validate the concentration parameters."""
        if not isinstance(self.alphas, np.ndarray):
            msg = "alphas must be a numpy ndarray."
            raise TypeError(msg)

        if self.alphas.ndim < 1:
            msg = "alphas must have at least one dimension."
            raise ValueError(msg)

        if np.any(self.alphas <= 0):
            msg = "alphas must be strictly positive."
            raise ValueError(msg)

        if self.alphas.shape[-1] < 2:
            msg = "Dirichlet distribution requires at least 2 classes."
            raise ValueError(msg)

    @classmethod
    def from_array(cls, alphas: np.ndarray | list, dtype: DTypeLike | None = None) -> Self:
        """Create a Dirichlet distribution from an array or list."""
        return cls(alphas=np.asarray(alphas, dtype=dtype))

    @override
    def sample(
        self,
        num_samples: int = 1,
        rng: np.random.Generator | None = None,
    ) -> ArraySample[ArrayCategoricalDistribution]:
        """Sample from the Dirichlet distribution (NumPy backend)."""
        if rng is None:
            rng = np.random.default_rng()

        gammas = rng.gamma(
            shape=self.alphas,
            scale=1.0,
            size=(num_samples, *self.alphas.shape),
        )

        samples = gammas / np.sum(gammas, axis=-1, keepdims=True)

        return ArraySample(array=ArrayCategoricalDistribution(samples), sample_axis=0)

    @override
    def _postprocess_ufunc_result(self, result: np.ndarray, *, ufunc: np.ufunc, method: str) -> np.ndarray:
        del ufunc, method
        return np.maximum(result, 1e-10)

    @override
    def __iter__(self) -> Iterator[Any]:
        return self.alphas.__iter__()

    @override
    def __eq__(self, value: Any) -> np.ndarray:  # ty: ignore[invalid-method-override]  # noqa: PYI032
        """Vectorized equality comparison."""
        if isinstance(value, ArrayDirichletDistribution):
            eq = np.equal(self.alphas, value.alphas)
        else:
            eq = np.equal(self.alphas, value)
        return np.all(eq, axis=-1)

    def __hash__(self) -> int:
        """Return an identity-based hash.

        We intentionally bypass ``super()`` here because protocol-heavy MROs can
        produce invalid ``super(type, obj)`` bindings at runtime. ``object``'s
        hash gives per-instance identity semantics.
        """
        return object.__hash__(self)
