"""Numpy-based distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self, override

import numpy as np

from probly.representation.distribution._common import CategoricalDistribution
from probly.representation.sample import ArraySample

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArrayCategoricalDistribution(
    CategoricalDistribution,
):
    """A categorical distribution stored as a numpy array.

    Shape: (..., num_classes)
    The last axis represents the category dimension.
    """

    probabilities: np.ndarray

    def __post_init__(self) -> None:
        """Validate the concentration parameters."""
        if not isinstance(self.probabilities, np.ndarray):
            msg = "probabilities must be a numpy ndarray."
            raise TypeError(msg)

        if self.probabilities.ndim < 1:
            msg = "probabilities must have at least one dimension."
            raise ValueError(msg)

        if np.any(self.probabilities < 0) or np.any(self.probabilities > 1):
            msg = "probabilities must be in the range [0, 1]."
            raise ValueError(msg)

        if not np.allclose(np.sum(self.probabilities, axis=-1), 1.0):
            msg = "probabilities must sum to 1."
            raise ValueError(msg)

    @property
    def num_classes(self) -> int:
        """Get the number of classes."""
        return self.probabilities.shape[-1]

    def __len__(self) -> int:
        """Return the length along the first dimension."""
        if self.ndim == 0:
            msg = "len() of unsized distribution"
            raise TypeError(msg)
        return len(self.probabilities)

    def __array_namespace__(self) -> Any:  # noqa: ANN401
        """Get the array namespace of the underlying array."""
        return self.probabilities.__array_namespace__()

    @property
    def dtype(self) -> DTypeLike:
        """The data type of the underlying array."""
        return self.probabilities.dtype

    @property
    def device(self) -> str:
        """The device of the underlying array."""
        return self.probabilities.device

    @property
    def ndim(self) -> int:
        """Number of batch dimensions (excluding category axis)."""
        return self.probabilities.ndim - 1

    @property
    def shape(self) -> tuple[int, ...]:
        """Batch shape (excluding category axis)."""
        return self.probabilities.shape[:-1]

    @property
    def size(self) -> int:
        """The total number of distributions."""
        return int(np.prod(self.shape)) if self.shape else 1

    @property
    def T(self) -> Self:  # noqa: N802
        """The transposed version of the distribution."""
        return np.transpose(self)  # ty: ignore[invalid-return-type]

    @property
    @override
    def entropy(self) -> float:
        """Compute the entropy of the categorical distribution."""
        p = self.probabilities
        return -np.sum(p * np.log(p), axis=-1)

    def sample(
        self,
        num_samples: int = 1,
        rng: np.random.Generator | None = None,
    ) -> ArraySample:
        """Sample from the categorical distribution (NumPy backend)."""
        if rng is None:
            rng = np.random.default_rng()

        samples = rng.choice(
            a=self.probabilities.shape[-1],
            size=(*self.shape, num_samples),
            p=self.probabilities,
        )

        return ArraySample(array=samples, sample_axis=0)

    def __setitem__(
        self,
        index: int | slice | tuple | np.ndarray,
        value: Self | np.ndarray,
    ) -> None:
        """Set a subset of the distribution by index."""
        if isinstance(value, ArrayCategoricalDistribution):
            self.probabilities[index] = value.probabilities
        else:
            self.probabilities[index] = value

    def __array__(
        self,
        dtype: DTypeLike | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        """Get the underlying numpy array (probabilities)."""
        if dtype is None and not copy:
            return self.probabilities
        return np.asarray(self.probabilities, dtype=dtype, copy=copy)

    def copy(self) -> Self:
        """Create a copy of the distribution."""
        return type(self)(probabilities=self.probabilities.copy())

    def to_device(self, device: Literal["cpu"]) -> Self:
        """Move the underlying array to the specified device."""
        if device == self.device:
            return self
        return type(self)(probabilities=self.probabilities.to_device(device))

    def __eq__(self, value: Any) -> Self:  # ty: ignore[invalid-method-override]  # noqa: ANN401, PYI032
        """Vectorized equality comparison."""
        if isinstance(value, ArrayCategoricalDistribution):
            return np.equal(self.probabilities, value.probabilities)  # ty: ignore[invalid-return-type]
        return np.equal(self.probabilities, value)

    def __hash__(self) -> int:
        """Compute the hash of the distribution."""
        return super().__hash__()
