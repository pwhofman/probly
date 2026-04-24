"""Numpy-based point prediction (degenerate/Dirac) distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, override

import numpy as np

from probly.representation._protected_axis.array import ArrayAxisProtected
from probly.representation.distribution._common import Distribution, DistributionSample
from probly.representation.sample.array import ArraySample


@dataclass(frozen=True, slots=True)
class ArrayPointPrediction(ArrayAxisProtected[np.ndarray], Distribution[np.ndarray]):
    """Deterministic point prediction stored as a numpy array.

    Minimal dispatch marker for ensemble members that output point predictions.
    Aleatoric uncertainty is zero by definition. All other behavior (stacking,
    indexing, shape) is inherited from ArrayAxisProtected.
    """

    mean: np.ndarray

    type: Literal["point_prediction"] = "point_prediction"
    protected_axes: ClassVar[dict[str, int]] = {"mean": 0}

    def __post_init__(self) -> None:
        """Coerce mean to float array."""
        object.__setattr__(self, "mean", np.asarray(self.mean, dtype=float))

    @override
    def sample(self, num_samples: int = 1, rng: np.random.Generator | None = None) -> ArraySample[np.ndarray]:
        """Return repeated copies of the point prediction."""
        del rng
        samples = np.broadcast_to(self.mean, (num_samples, *self.mean.shape))
        return ArraySample(array=samples.copy(), sample_axis=0)


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
