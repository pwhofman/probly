"""Torch sample implementation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import torch

from probly.representation.sampling.common_sample import Sample

from .sample import create_sample

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import DTypeLike

    from probly.representation.sampling.common_sample import SampleAxis

type Numeric = torch.Tensor | float | int


@create_sample.register(torch.Tensor)
class TorchTensorSample(Sample[torch.Tensor]):
    """A sample implementation for torch tensors."""

    sample_axis: int
    array: torch.Tensor

    def __init__(self, samples: list[torch.Tensor], sample_axis: int) -> None:
        """Initialize the torch tensor sample."""
        self.array = torch.stack(samples).permute(1, 0, 2)  # we use the convention [instances, samples, classes]
        self.sample_axis = sample_axis

    def from_iterable(
        self,
        samples: Iterable[Numeric],
        sample_axis: SampleAxis = "auto",
        dtype: DTypeLike = None,
    ) -> TorchTensorSample:
        """Create an TorchTensorSample from a sequence of samples.

        Args:
            samples: The predictions to create the sample from.
            sample_axis: The dimension along which samples are organized.
            dtype: Desired data type of the array.

        Returns:
            The created TorchTensorSample.
        """
        if isinstance(samples, torch.Tensor):
            if sample_axis == "auto":
                if samples.ndim == 0:
                    msg = "Cannot infer sample_axis for 0-dimensional array."
                    raise ValueError(msg)
                sample_axis = 0 if samples.ndim == 1 else 1
            if sample_axis != 0:
                samples = torch.moveaxis(samples, 0, sample_axis)
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
                sample_axis = (0 if first_sample.ndim == 0 else 1) if isinstance(first_sample, torch.Tensor) else 0
            samples = torch.stack(samples, dim=sample_axis)

        return self(array=samples, sample_axis=sample_axis)

    def samples(self) -> torch.Tensor:
        """Return an iterator over the samples."""
        if self.sample_axis == 0:
            return self.array
        return torch.moveaxis(self.array, self.sample_axis, 0)

    def sample_mean(self) -> torch.Tensor:
        """Compute the mean of the sample."""
        return self.array.mean(dim=self.sample_axis)

    def sample_std(self, ddof: int = 1) -> torch.Tensor:
        """Compute the standard deviation of the sample."""
        return self.array.std(dim=self.sample_axis, correction=ddof)

    def sample_var(self, ddof: int = 1) -> torch.Tensor:
        """Compute the variance of the sample."""
        return self.array.var(dim=self.sample_axis, correction=ddof)
