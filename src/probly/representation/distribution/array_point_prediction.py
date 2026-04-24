"""Numpy-based point prediction (degenerate/Dirac) distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, override

import numpy as np

from probly.representation._protected_axis.array import ArrayAxisProtected
from probly.representation.distribution._common import Distribution, DistributionSample
from probly.representation.sample.array import ArraySample

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import DTypeLike


@dataclass(frozen=True, slots=True, weakref_slot=True)
class ArrayPointPrediction(ArrayAxisProtected[np.ndarray], Distribution[np.ndarray]):
    """Deterministic point prediction stored as a numpy array.

    Represents ensemble members that output point predictions rather than
    distributional predictions. Has a mean but no variance -- aleatoric
    uncertainty is zero by definition.
    """

    mean: np.ndarray

    type: Literal["point_prediction"] = "point_prediction"
    protected_axes: ClassVar[dict[str, int]] = {"mean": 0}
    allowed_types: ClassVar[tuple[type[np.ndarray] | type[np.generic] | type[float] | type[int], ...]] = (
        np.ndarray,
        np.generic,
        float,
        int,
    )

    def __post_init__(self) -> None:
        """Validate and coerce mean to float array."""
        mean = np.asarray(self.mean, dtype=float)
        object.__setattr__(self, "mean", mean)

    @override
    def sample(
        self,
        num_samples: int = 1,
        rng: np.random.Generator | None = None,
    ) -> ArraySample[np.ndarray]:
        """Return repeated copies of the point prediction."""
        del rng
        samples = np.broadcast_to(self.mean, (num_samples, *self.mean.shape))
        return ArraySample(array=samples.copy(), sample_axis=0)

    @override
    def __array__(
        self,
        dtype: DTypeLike | None = None,
        /,
        *,
        copy: bool | None = None,
    ) -> np.ndarray:
        """Represent the point prediction as its mean array."""
        return np.asarray(self.mean, dtype=dtype, copy=copy)

    @override
    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: object,
        **kwargs: object,
    ) -> Any:
        """Arithmetical operations for point predictions."""
        if ufunc is not np.add or method != "__call__":
            return NotImplemented

        out = kwargs.get("out", ())
        for x in (*inputs, *out):
            if not isinstance(x, (*self.allowed_types, type(self))):
                return NotImplemented

        unpacked: list[np.ndarray | float | int] = []
        for x in inputs:
            if isinstance(x, type(self)):
                unpacked.append(x.mean)
            else:
                unpacked.append(x)

        new_mean = ufunc(*unpacked, **{k: v for k, v in kwargs.items() if k != "out"})
        result = type(self)(mean=np.asarray(new_mean))

        if kwargs.get("out"):
            out_obj = kwargs["out"][0]
            if isinstance(out_obj, type(self)):
                object.__setattr__(out_obj, "mean", result.mean)
                return out_obj
            return NotImplemented

        return result

    @override
    def __iter__(self) -> Iterator[Any]:
        return self.mean.__iter__()

    @override
    def __eq__(self, other: Any) -> np.ndarray:  # ty: ignore[invalid-method-override]  # noqa: PYI032
        """Compare two point predictions by their means."""
        if not isinstance(other, ArrayPointPrediction):
            return NotImplemented
        return np.equal(self.mean, other.mean)

    def __hash__(self) -> int:
        """Return an identity-based hash.

        We intentionally bypass ``super()`` here because protocol-heavy MROs can
        produce invalid ``super(type, obj)`` bindings at runtime. ``object``'s
        hash gives per-instance identity semantics.
        """
        return object.__hash__(self)


class ArrayPointPredictionSample(  # ty:ignore[conflicting-metaclass]
    DistributionSample[ArrayPointPrediction],
    ArraySample[ArrayPointPrediction],
):
    """Sample type for empirical second-order point prediction distributions."""

    sample_space: ClassVar[type[Distribution]] = ArrayPointPrediction

    @override
    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        return super().__instancehook__(instance)
