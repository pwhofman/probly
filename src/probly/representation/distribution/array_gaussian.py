"""Numpy-based Gaussian distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self, cast, override

from matplotlib.pylab import ArrayLike
import numpy as np

from probly.representation.distribution.common import GaussianDistribution
from probly.representation.sampling.array_sample import ArraySample

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike

_rng = np.random.default_rng()


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArrayGaussian(GaussianDistribution):
    """Gaussian distribution with array parameters."""

    mean: np.ndarray
    var: np.ndarray

    type: Literal["gaussian"] = "gaussian"

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

    @property
    @override
    def entropy(self) -> float:
        """Return the total differential entropy of the Gaussian distribution."""
        var = self.var
        return float(0.5 * np.log(2 * np.e * np.pi * var).sum())

    @override
    def sample(self, size: int) -> ArraySample[np.ndarray]:
        """Draw samples and wrap them in an ArraySample (sample_axis=0)."""
        std = np.sqrt(self.var)
        samples = _rng.normal(
            loc=self.mean,
            scale=std,
            size=(size, *self.mean.shape),
        )
        return ArraySample(array=samples, sample_axis=0)

    def __array_namespace__(self) -> object:
        """Return the array namespace used by this distribution (NumPy)."""
        return self.mean.__array_namespace__()

    def copy(self) -> Self:
        """Create a copy of the gaussian distribution."""
        return type(self)(
            mean=self.mean.copy(),
            var=self.var.copy(),
        )

    def __array__(
        self,
        dtype: DTypeLike | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        """Represent the distribution as stacked [mean, var] on the last axis."""
        stacked = np.stack([self.mean, self.var], axis=-1)
        if dtype is None and not copy:
            return stacked
        return np.asarray(stacked, dtype=dtype, copy=copy)

    @property
    def ndim(self) -> int:
        """The number of dimensions of the underlying array."""
        return int(self.mean.ndim)

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the underlying array."""
        return tuple(self.mean.shape)

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
    def device(self) -> str:
        """Return the hardware device on which the arrays reside (CPU for NumPy)."""
        return cast("str", self.mean.device)

    def __getitem__(self, index: int | slice | tuple | np.ndarray) -> Self:
        """Return a sliced view of this Gaussian."""
        return type(self)(
            mean=self.mean[index],
            var=self.var[index],
        )

    def __len__(self) -> int:
        """Return the length of the underlying mean array."""
        return len(self.mean)

    def __repr__(self) -> str:
        """Return a debug representation of the Gaussian."""
        cls_name = type(self).__name__
        return f"{cls_name}(mean={self.mean!r}, var={self.var!r}, type={self.type!r})"

    def __eq__(self, other: object) -> bool:
        """Compare two Gaussians by their parameters."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.array_equal(self.mean, other.mean) and np.array_equal(self.var, other.var) and self.type == other.type

    def __hash__(self) -> int:
        """Compute the hash of the distribution."""
        return super().__hash__()  # type: ignore[no-any-return]
