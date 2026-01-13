"""Sklearn sample implementation."""

from __future__ import annotations

import numpy as np

from .sample import Sample, create_sample
from collections.abc import Sequence

from typing import TYPE_CHECKING, Self
from probly.representation.sampling.common_sample import Sample, SampleAxis

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import DTypeLike

type Numeric = np.number | np.ndarray | float | int

@create_sample.register(np.ndarray)
class SklearnSample(Sample[np.ndarray]):
    """A sample implementation for scikit-learn models."""

    def from_iterable(cls, samples: Iterable[Numeric], sample_axis: SampleAxis = "auto", dtype: DTypeLike = None) -> Self:
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

        return cls(array=samples, sample_axis=sample_axis)
    
    def samples(cls, sample: Sample[Numeric], sample_axis: SampleAxis = "auto", dtype: DTypeLike = None) -> Iterable[np.ndarray]:
        """Return an iterator over the samples."""
        if sample_axis == "auto":
            sample_axis = 1
        for i in range(sample.array.shape[sample_axis]):
            yield np.take(sample.array, indices=i, axis=sample_axis)

    def __init__(self, samples: list[np.ndarray], sample_axis: int = 1) -> None:
        """Initialize the sample."""
        self.array = np.stack(samples, axis=sample_axis)

    def sample_mean(self, sample_axis: int = 1) -> np.ndarray:
        """Compute mean across the samples axis."""
        return self.array.mean(axis=sample_axis) 

    def sample_std(self, ddof: int = 1, sample_axis: int = 1) -> np.ndarray:
        """Compute standard deviation across the samples axis."""
        return self.array.std(axis=sample_axis, ddof=ddof)

    def sample_var(self, ddof: int = 1, sample_axis: int = 1) -> np.ndarray:
        """Compute variance across the samples axis."""
        return self.array.var(axis=sample_axis, ddof=ddof) 