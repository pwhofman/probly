"""Classes representing credal sets."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, override

import torch

from probly.representation.credal_set.common import (
    CategoricalCredalSet,
    ProbabilityIntervalsCredalSet,
)
from probly.representation.sampling.torch_sample import TorchTensorSample

from .common import create_probability_intervals

if TYPE_CHECKING:
    from probly.representation.sampling.common_sample import Sample


class TorchCategoricalCredalSet(CategoricalCredalSet[torch.Tensor], metaclass=ABCMeta):
    """A credal set of predictions stored in a numpy array."""

    @override
    @classmethod
    def from_sample(cls, sample: Sample[torch.Tensor], distribution_axis: int = -1) -> Self:
        array_sample = TorchTensorSample.from_sample(sample)
        return cls.from_torch_sample(array_sample, distribution_axis=distribution_axis)

    @classmethod
    @abstractmethod
    def from_torch_sample(
        cls,
        sample: TorchTensorSample,
        distribution_axis: int = -1,
    ) -> Self:
        """Create a credal set from an ArraySample.

        Args:
            sample: The sample to create the credal set from.
            distribution_axis: The axis in each sample containing the categorical probabilities.

        Returns:
            The created credal set.
        """
        msg = "from_array_sample method not implemented."
        raise NotImplementedError(msg)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchProbabilityIntervals(TorchCategoricalCredalSet, ProbabilityIntervalsCredalSet[torch.Tensor]):
    """A credal set defined by probability intervals over outcomes.

    This represents uncertainty through lower and upper probability bounds for each class.
    Each bound is stored as a seperate numpy array of shape (..., num_classes).
    """

    lower_bounds: torch.Tensor
    upper_bounds: torch.Tensor

    @override
    @classmethod
    def from_torch_sample(
        cls,
        sample: TorchTensorSample,
        distribution_axis: int = -1,
    ) -> Self:
        """Create probability intervals from a sample by computing min/max bounds.

        Args:
            sample: The sample to extract intervals from.
            distribution_axis: Which axis contains the categorical probabilities.

        Returns:
            A new ArrayProbabilityIntervals instance.
        """
        if distribution_axis < 0:
            distribution_axis += sample.ndim - 1

        # Get all samples in shape (..., num_samples, num_classes)
        samples_array = torch.moveaxis(sample.samples, distribution_axis + 1, -1)

        # Compute lower and upper bounds across samples
        lower_bounds = torch.min(samples_array, dim=-2)[0]
        upper_bounds = torch.max(samples_array, dim=-2)[0]

        return cls(lower_bounds=lower_bounds, upper_bounds=upper_bounds)

    @property
    def device(self) -> str | torch.device:
        """Return the device where the bounds are stored."""
        return self.lower_bounds.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the data type of the bounds."""
        return self.lower_bounds.dtype

    @property
    def ndim(self) -> int:
        """Return the number of dimensions (excluding the class dimensions)."""
        return self.lower_bounds.ndim - 1

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape (excluding the class dimensions)."""
        return self.lower_bounds.shape[:-1]

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return self.lower_bounds.shape[-1]

    def __len__(self) -> int:
        """Return the length of the first dimension."""
        shape = self.shape

        if len(shape) == 0:
            msg = "len() of unsized credal set"
            raise TypeError(msg)

        return shape[0]

    def __array__(self, dtype: torch.dtype | None = None, copy: bool | None = None) -> torch.Tensor:
        """Get the intervals as a stacked array with shape (..., 2, num_classes).

        Args:
            dtype: Desired data type.
            copy: Whether to return a copy.

        Returns:
            Stacked array of [lower_bounds, upper_bounds].
        """
        stacked = torch.stack([self.lower_bounds, self.upper_bounds], dim=-2)

        if dtype is None and not copy:
            return stacked

        if copy:
            return torch.tensor(stacked, dtype=dtype).clone()

        return torch.tensor(stacked, dtype=dtype)

    @override
    def lower(self) -> torch.Tensor:
        """Get the lower probability bounds for each class."""
        return self.lower_bounds

    @override
    def upper(self) -> torch.Tensor:
        """Get the upper probability bounds for each class."""
        return self.upper_bounds

    def width(self) -> torch.Tensor:
        """Compute the width of each probability interval.

        Returns:
            Array of interval widths for each class.
        """
        return self.upper_bounds - self.lower_bounds

    def contains(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Check if given probabilities fall within the intervals.

        Args:
            probabilities: Probability distributions to check, shape (..., num_classes).

        Returns:
            Boolean array indicating whether each probability is contained.
        """
        within_bounds = (probabilities >= self.lower_bounds) & (probabilities <= self.upper_bounds)
        return torch.all(within_bounds, dim=-1)

    def clone(self) -> Self:
        """Create a copy of the intervals.

        Returns:
            A new ArrayProbabilityIntervals with copied data.
        """
        return type(self)(
            lower_bounds=self.lower_bounds.clone(),
            upper_bounds=self.upper_bounds.clone(),
        )

    def to(self, device: str | torch.device) -> Self:
        """Move the intervals to a specified device.

        Args:
            device: Target device.

        Returns:
            A new ArrayProbabilityIntervals on the specified device.
        """
        if device == self.device:
            return self

        return type(self)(
            lower_bounds=self.lower_bounds.to(device),
            upper_bounds=self.upper_bounds.to(device),
        )

    def __eq__(self, value: Any) -> Self:  # type: ignore[override]  # noqa: ANN401, PYI032
        """Vectorized equality comparison."""
        return torch.equal(self, value)  # type: ignore[return-value]

    def __hash__(self) -> int:
        """Compute the hash of the intervals."""
        return super().__hash__()


create_probability_intervals.register(torch.Tensor, TorchProbabilityIntervals.from_sample)
