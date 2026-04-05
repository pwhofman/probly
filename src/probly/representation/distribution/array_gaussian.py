"""Numpy-based Gaussian distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self, cast, overload, override

import numpy as np

from probly.representation.array_like import NumpyArrayLike
from probly.representation.distribution._common import GaussianDistribution
from probly.representation.sample.array import ArraySample

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import ModuleType

    from numpy.typing import ArrayLike, DTypeLike

    from probly.representation.array_like import ArrayFlagsLike, ToIndices


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArrayGaussianDistribution(NumpyArrayLike[Any], GaussianDistribution[Any]):
    """Gaussian distribution with array parameters."""

    mean: np.ndarray
    var: np.ndarray

    type: Literal["gaussian"] = "gaussian"

    allowed_types: tuple[type[np.ndarray] | type[np.generic] | type[float] | type[int], ...] = (
        np.ndarray,
        np.generic,
        float,
        int,
    )

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
    ) -> ArrayGaussianDistribution:
        """Create an ArrayGaussian from mean and variance parameters."""
        mean_arr = np.asarray(mean, dtype=dtype if dtype is not None else float)
        var_arr = np.asarray(var, dtype=mean_arr.dtype)
        return cls(mean=mean_arr, var=var_arr)

    @override
    @property
    def entropy(self) -> np.ndarray:
        """Return the total differential entropy of the Gaussian distribution."""
        var = self.var
        return 0.5 * np.log(2 * np.e * np.pi * var)

    @override
    def sample(
        self,
        num_samples: int = 1,
        rng: np.random.Generator | None = None,
    ) -> ArraySample[np.ndarray]:
        """Draw samples and wrap them in an ArraySample (sample_axis=0)."""
        if rng is None:
            rng = np.random.default_rng()

        std = np.sqrt(self.var)
        samples = rng.normal(
            loc=self.mean,
            scale=std,
            size=(num_samples, *self.mean.shape),
        )
        return ArraySample(array=cast("NumpyArrayLike[Any]", samples), sample_axis=0)

    @override
    def __array_namespace__(self, /, *, api_version: str | None = None) -> ModuleType:
        """Return the array namespace used by this distribution (NumPy)."""
        del api_version
        return np

    @override
    @property
    def flags(self) -> ArrayFlagsLike:
        return self.mean.flags

    @overload
    def __array__(self) -> np.ndarray: ...

    @overload
    def __array__(self, dtype: DTypeLike) -> np.ndarray: ...

    @override
    def __array__(
        self,
        dtype: DTypeLike | None = None,
        /,
        *,
        copy: bool | None = None,
    ) -> np.ndarray:
        """Represent the distribution as stacked [mean, var] on the last axis."""
        stacked = np.stack([self.mean, self.var], axis=-1)
        return np.asarray(stacked, dtype=dtype, copy=copy)

    @override
    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: object,
        **kwargs: object,
    ) -> Any:
        """Arithmetical operations for Gaussian."""
        if ufunc is not np.add or method != "__call__":  # just + for now
            return NotImplemented

        out = kwargs.get("out", ())

        for x in (*inputs, *out):
            if not isinstance(x, (*self.allowed_types, type(self))):
                return NotImplemented

        unpacked: list[np.ndarray | float | int] = []
        gaussians: list[ArrayGaussianDistribution] = []

        for x in inputs:
            if isinstance(x, type(self)):
                gaussians.append(x)
                unpacked.append(x.mean)
            else:
                unpacked.append(x)

        if not gaussians:
            return NotImplemented

        new_mean = ufunc(*unpacked, **{k: v for k, v in kwargs.items() if k != "out"})

        new_var = np.zeros_like(gaussians[0].var, dtype=float)
        for g in gaussians:
            new_var = new_var + g.var

        result = type(self)(mean=np.asarray(new_mean), var=np.asarray(new_var))

        if kwargs.get("out"):
            out_gaussian = kwargs["out"][0]
            if isinstance(out_gaussian, type(self)):
                object.__setattr__(out_gaussian, "mean", result.mean)
                object.__setattr__(out_gaussian, "var", result.var)
                return out_gaussian
            return NotImplemented

        return result

    @override
    def __array_function__(
        self,
        func: Callable,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Handle numpy array functions."""
        if not all(issubclass(t, (type(self), np.ndarray)) for t in types):
            return NotImplemented

        result: Any
        if func is np.copy:
            subok = kwargs.get("subok", True)
            order = kwargs.get("order", "C")
            if not subok:
                result = np.copy(np.asarray(self), order=order, subok=subok)
            else:
                result = type(self)(mean=np.copy(self.mean, order=order), var=np.copy(self.var, order=order))
        elif func is np.transpose:
            axes = kwargs.get("axes")
            result = type(self)(mean=np.transpose(self.mean, axes=axes), var=np.transpose(self.var, axes=axes))
        elif func is np.matrix_transpose:
            result = type(self)(mean=np.matrix_transpose(self.mean), var=np.matrix_transpose(self.var))
        elif func is np.astype:
            dtype = kwargs["dtype"]
            copy = kwargs.get("copy", True)
            result = type(self)(
                mean=np.astype(self.mean, dtype=dtype, copy=copy),
                var=np.astype(self.var, dtype=dtype, copy=copy),
            )
        else:
            return NotImplemented

        return result

    @override
    @property
    def ndim(self) -> int:
        """The number of dimensions of the underlying array."""
        return self.mean.ndim

    @override
    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the underlying array."""
        return self.mean.shape

    @override
    @property
    def size(self) -> int:
        """The total number of elements in the underlying array."""
        return self.mean.size

    @override
    @property
    def dtype(self) -> DTypeLike:
        """The data type of the underlying array."""
        return self.mean.dtype

    @override
    @property
    def device(self) -> str:
        """Return the hardware device on which the arrays reside (CPU for NumPy)."""
        return self.mean.device

    @override
    def __getitem__(self, index: ToIndices, /) -> Self:
        """Return a sliced view of this Gaussian."""
        return type(self)(
            mean=self.mean[index],
            var=self.var[index],
        )

    @override
    def __setitem__(self, index: ToIndices, value: object, /) -> None:
        """Set a subset of the Gaussian by index."""
        if isinstance(value, ArrayGaussianDistribution):
            self.mean[index] = value.mean
            self.var[index] = value.var
            return

        if isinstance(value, tuple) and len(value) == 2:
            self.mean[index], self.var[index] = value
            return

        msg = "value must be an ArrayGaussianDistribution or a (mean, var) tuple."
        raise TypeError(msg)

    @override
    def __len__(self) -> int:
        """Return the length of the underlying mean array."""
        return len(self.mean)

    @override
    def __iter__(self) -> Iterator[Any]:
        return self.mean.__iter__()

    def __eq__(self, other: object) -> bool:
        """Compare two Gaussians by their parameters."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.array_equal(self.mean, other.mean) and np.array_equal(self.var, other.var) and self.type == other.type

    def __hash__(self) -> int:
        """Compute the hash of the distribution."""
        return super().__hash__()
