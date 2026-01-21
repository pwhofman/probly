"""Numpy-based distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self, override

import numpy as np
from scipy import special

from probly.representation.distribution.common import DirichletDistribution

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArrayDirichletDistribution(
    DirichletDistribution,
    np.lib.mixins.NDArrayOperatorsMixin,
):
    """A Dirichlet distribution stored as a numpy array.

    Shape: (..., num_classes)
    The last axis represents the category dimension.
    """

    alphas: np.ndarray

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
    def from_array(cls, alphas: np.ndarray | list, dtype: DTypeLike = None) -> Self:
        """Create a Dirichlet distribution from an array or list."""
        return cls(alphas=np.asarray(alphas, dtype=dtype))

    def __len__(self) -> int:
        """Return the length along the first dimension."""
        if self.ndim == 0:
            msg = "len() of unsized distribution"
            raise TypeError(msg)
        return len(self.alphas)

    def __array_namespace__(self) -> Any:  # noqa: ANN401
        """Get the array namespace of the underlying array."""
        return self.alphas.__array_namespace__()

    @property
    def dtype(self) -> DTypeLike:
        """The data type of the underlying array."""
        return self.alphas.dtype  # type: ignore[no-any-return]

    @property
    def device(self) -> str:
        """The device of the underlying array."""
        return self.alphas.device  # type: ignore[no-any-return]

    @property
    def ndim(self) -> int:
        """Number of batch dimensions (excluding category axis)."""
        return self.alphas.ndim - 1  # type: ignore[no-any-return]

    @property
    def shape(self) -> tuple[int, ...]:
        """Batch shape (excluding category axis)."""
        return self.alphas.shape[:-1]  # type: ignore[no-any-return]

    @property
    def size(self) -> int:
        """The total number of distributions."""
        return int(np.prod(self.shape)) if self.shape else 1

    @property
    def T(self) -> Self:  # noqa: N802
        """The transposed version of the distribution."""
        return np.transpose(self)  # type: ignore[no-any-return]

    @property
    @override
    def entropy(self) -> float:
        """Compute the entropy of the Dirichlet distribution."""
        alpha_0 = np.sum(self.alphas, axis=-1)
        K = self.alphas.shape[-1]  # noqa: N806

        log_beta = np.sum(special.gammaln(self.alphas), axis=-1) - special.gammaln(alpha_0)
        digamma_sum = (alpha_0 - K) * special.digamma(alpha_0)
        digamma_individual = np.sum((self.alphas - 1) * special.digamma(self.alphas), axis=-1)

        return log_beta + digamma_sum - digamma_individual  # type: ignore[no-any-return]

    def __setitem__(
        self,
        index: int | slice | tuple | np.ndarray,
        value: Self | np.ndarray,
    ) -> None:
        """Set a subset of the distribution by index."""
        if isinstance(value, ArrayDirichletDistribution):
            self.alphas[index] = value.alphas
        else:
            self.alphas[index] = value

    def __array__(
        self,
        dtype: DTypeLike = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        """Get the underlying numpy array (alphas)."""
        if dtype is None and not copy:
            return self.alphas
        return np.asarray(self.alphas, dtype=dtype, copy=copy)

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        """Handle numpy ufuncs."""
        alphas_inputs = [x.alphas if isinstance(x, ArrayDirichletDistribution) else x for x in inputs]

        if method in ("__call__", "reduce", "reduceat", "accumulate") and "out" in kwargs:
            outs = kwargs["out"]
            if outs is not None:
                if not isinstance(outs, tuple):
                    outs = (outs,)
                kwargs["out"] = tuple(o.alphas if isinstance(o, ArrayDirichletDistribution) else o for o in outs)
        else:
            outs = None

        result = getattr(ufunc, method)(*alphas_inputs, **kwargs)

        if outs is not None:
            return outs[0] if len(outs) == 1 else outs

        if isinstance(result, np.ndarray) and result.ndim > 0:
            result = np.maximum(result, 1e-10)
            return type(self)(alphas=result)

        return result

    def copy(self) -> Self:
        """Create a copy of the distribution."""
        return type(self)(alphas=self.alphas.copy())

    def to_device(self, device: Literal["cpu"]) -> Self:
        """Move the underlying array to the specified device."""
        if device == self.device:
            return self
        return type(self)(alphas=self.alphas.to_device(device))

    def __eq__(self, value: Any) -> Self:  # type: ignore[override]  # noqa: ANN401, PYI032
        """Vectorized equality comparison."""
        if isinstance(value, ArrayDirichletDistribution):
            return np.equal(self.alphas, value.alphas)  # type: ignore[no-any-return]
        return np.equal(self.alphas, value)  # type: ignore[no-any-return]

    def __hash__(self) -> int:
        """Compute the hash of the distribution."""
        return super().__hash__()  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        """Return a string representation of the distribution."""
        return f"ArrayDirichletDistribution(alphas={self.alphas})"
