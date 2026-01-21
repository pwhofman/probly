"""Numpy-based Gaussian distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np

from probly.representation.distribution.common import Distribution, DistributionType

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArrayGaussian(Distribution):
    """Gaussian distribution with array parameters."""

    mean: np.ndarray
    var: np.ndarray

    type: DistributionType = "gaussian"

    def __post_init__(self) -> None:
        """Validate shapes and variances."""
        mean = np.asarray(self.mean, dtype=float)
        var = np.asarray(self.var, dtype=float)

        if mean.shape != var.shape:
            msg = f"mean and var must have same shape, got {mean.shape} and {var.shape}"
            raise ValueError(msg)
        if np.any(var <= 0):
            var_error = "Variance must be positive"
            raise ValueError(var_error)

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "var", var)

    @property
    def entropy(self) -> float:
        """Return the total differential entropy of the Gaussian distribution."""
        var = self.var
        return float(0.5 * np.log(2 * np.e * np.pi * var).sum())

    @classmethod
    def from_parameters(
        cls,
        mean: ArrayLike,
        var: ArrayLike,
        dtype: DTypeLike | None = None,
    ) -> ArrayGaussian:
        """Create an ArrayGaussian from mean and variance parameters."""
        mean_arr = np.asarray(mean, dtype=dtype if dtype is not None else float)
        var_arr = np.asarray(var, dtype=mean_arr.dtype)

        return cls(mean=mean_arr, var=var_arr)

    '''def sample(self, size: int) -> ArraySample[np.ndarray]:
        """Draw samples and wrap them in an ArraySample.

        Returns an ArraySample with sample_axis=0.
        """
        std = np.sqrt(self.var)

        samples = np.random.normal(
            loc=self.mean,
            scale=std,
            size=(size,) + self.mean.shape,
        )
        return ArraySample(array=samples, sample_axis=0)'''

    @property
    def __array_namespace__(self) -> object:
        """Return the array namespace used by this distribution (NumPy)."""
        return np

    @property
    def ndim(self) -> int:
        """The number of dimensions of the underlying array."""
        return int(self.mean.ndim)

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the underlying array."""
        return tuple[int, ...](self.mean.shape)

    @property
    def size(self) -> int:
        """The total number of elements in the underlying array."""
        return int(self.mean.size)

    @property
    def T(self) -> Self:  # noqa: N802
        """Return a new ArrayGaussian with transposed parameters."""
        return type(self)(
            mean=np.transpose(self.mean),
            var=np.transpose(self.var),
        )

    @property
    def dtype(self) -> DTypeLike:
        """The data type of the underlying array."""
        return self.mean.dtype

    @property
    def mT(self) -> Self:  # noqa: N802
        """Return a new ArrayGaussian with matrix-transposed parameters."""
        mean_t = np.matrix_transpose(self.mean)
        var_t = np.matrix_transpose(self.var)
        return type(self)(mean=mean_t, var=var_t)

    @property
    def device(self) -> str:
        """Return the hardware device on which the arrays reside (CPU for NumPy)."""
        return "cpu"
