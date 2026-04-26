"""Numpy-based Gaussian distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, override

import numpy as np

from probly.representation._protected_axis.array import ArrayAxisProtected
from probly.representation.distribution._common import GaussianDistribution, GaussianDistributionSample
from probly.representation.sample.array import ArraySample

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import DTypeLike


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArrayGaussianDistribution(ArrayAxisProtected[np.ndarray], GaussianDistribution[np.ndarray]):
    """Gaussian distribution with array parameters."""

    mean: np.ndarray
    var: np.ndarray

    type: Literal["gaussian"] = "gaussian"
    protected_axes: ClassVar[dict[str, int]] = {"mean": 0, "var": 0}
    allowed_types: ClassVar[tuple[type[np.ndarray] | type[np.generic] | type[float] | type[int], ...]] = (
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
            msg = "Variance must be positive"
            raise ValueError(msg)

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "var", var)

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
        return ArraySample(array=samples, sample_axis=0)

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
        """Arithmetical operations for Gaussian distributions."""
        if ufunc is not np.add or method != "__call__":
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
    def __iter__(self) -> Iterator[Any]:
        return self.mean.__iter__()

    @override
    def __eq__(self, other: Any) -> np.ndarray:  # ty: ignore[invalid-method-override]  # noqa: PYI032
        """Compare two Gaussians by their parameters."""
        if not isinstance(other, ArrayGaussianDistribution):
            return NotImplemented
        return np.equal(self.mean, other.mean) & np.equal(self.var, other.var)

    def __hash__(self) -> int:
        """Return an identity-based hash.

        We intentionally bypass ``super()`` here because protocol-heavy MROs can
        produce invalid ``super(type, obj)`` bindings at runtime. ``object``'s
        hash gives per-instance identity semantics.
        """
        return object.__hash__(self)


class ArrayGaussianDistributionSample(  # ty:ignore[conflicting-metaclass]
    GaussianDistributionSample[ArrayGaussianDistribution],
    ArraySample[ArrayGaussianDistribution],
):
    """Sample type for empirical second-order Gaussian distributions."""

    sample_space: ClassVar[type[GaussianDistribution]] = ArrayGaussianDistribution

    @override
    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        return super().__instancehook__(instance)
