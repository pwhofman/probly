"""Sklearn sample implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

import numpy as np

from probly.representation.sampling.common_sample import Sample

from .sample import create_sample

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import DTypeLike

    from probly.representation.sampling.common_sample import SampleAxis

type Numeric = np.number | np.ndarray | float | int


@create_sample.register(np.ndarray)
class SklearnSample(Sample[np.ndarray]):
    """A sample implementation for scikit-learn models."""

    sample_axis: int
    array: np.ndarray

    def from_iterable(
        self,
        samples: Iterable[Numeric],
        sample_axis: SampleAxis = "auto",
        dtype: DTypeLike = None,
    ) -> Self:
        """Create an ArraySample for Sklearn from a sequence of samples.

        Args:
            samples: The predictions to create the sample from.
            sample_axis: The dimension along which samples are organized.
            dtype: Desired data type of the array.

        Returns:
            The created ArraySample for sklearn.
        """
        if isinstance(samples, np.ndarray):
            if sample_axis == "auto":
                if samples.ndim == 0:
                    msg = "Cannot infer sample_axis for 0-dimensional array."
                    raise ValueError(msg)
                sample_axis = 0 if samples.ndim == 1 else 1
            if sample_axis != 0:
                samples = np.moveaxis(samples, 0, sample_axis)
            if dtype is not None:
                samples = samples.astype(dtype)
        else:
            if not isinstance(samples, Sequence):
                samples = list(samples)
            if sample_axis == "auto":
                if len(samples) == 0:
                    msg = "Cannot infer sample_axis for empty samples."
                    raise ValueError(msg)
                first_sample = samples[0]
                sample_axis = (0 if first_sample.ndim == 0 else 1) if isinstance(first_sample, np.ndarray) else 0
            samples = np.stack(samples, axis=sample_axis, dtype=dtype)

        return self(array=samples, sample_axis=sample_axis)

    def samples(self) -> Iterable[np.ndarray]:
        """Return an iterator over the samples."""
        if self.sample_axis == 0:
            return self.array
        return np.moveaxis(self.array, self.sample_axis, 0)

    def __init__(self, samples: list[np.ndarray], sample_axis: int = 1) -> None:
        """Initialize the sample."""
        self.array = np.stack(samples, axis=sample_axis)
        self.sample_axis = sample_axis

    def sample_mean(self) -> np.ndarray:
        """Compute mean across the samples axis."""
        return self.array.mean(axis=self.sample_axis)

    def sample_std(self, ddof: int = 1) -> np.ndarray:
        """Compute standard deviation across the samples axis."""
        return self.array.std(axis=self.sample_axis, ddof=ddof)

    def sample_var(self, ddof: int = 1) -> np.ndarray:
        """Compute variance across the samples axis."""
        return self.array.var(axis=self.sample_axis, ddof=ddof)
