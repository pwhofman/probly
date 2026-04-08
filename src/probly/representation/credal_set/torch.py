"""Classes representing credal sets."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, override

import numpy as np
import torch

from probly.representation.credal_set import create_convex_credal_set
from probly.representation.credal_set._common import (
    CategoricalCredalSet,
    ConvexCredalSet,
    ProbabilityIntervalsCredalSet,
)
from probly.representation.sample.torch import TorchTensorSample

from ._common import create_probability_intervals

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from probly.representation.sample._common import Sample


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
class TorchConvexCredalSet(TorchCategoricalCredalSet, ConvexCredalSet[torch.Tensor]):
    """A convex credal set defined by the convex hull of distributions stored in a torch tensor.

    Internally, this is represented exactly like a discrete credal set:
    an array of shape (..., num_vertices, num_classes), where the distributions
    are the extreme points (vertices) of the polytope.
    """

    tensor: torch.Tensor

    @override
    @classmethod
    def from_torch_sample(
        cls,
        sample: TorchTensorSample,
        distribution_axis: int = -1,
    ) -> Self:
        if distribution_axis < 0:
            distribution_axis += sample.ndim - 1

        tensor = torch.moveaxis(sample.samples, (0, distribution_axis + 1), (-2, -1))

        return cls(tensor=tensor)

    @override
    @classmethod
    def from_data(cls, data: torch.Tensor, distribution_axis: int = -1) -> Self:
        if distribution_axis < 0:
            distribution_axis += data.ndim - 1

        tensor = torch.moveaxis(data, distribution_axis, -2)

        return cls(tensor=tensor)

    @property
    def device(self) -> str | torch.device:
        """Return the device of the credal set array."""
        return self.tensor.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the data type of the credal set array."""
        return self.tensor.dtype

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the credal set array."""
        return self.tensor.ndim - 2

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the credal set array."""
        return self.tensor.shape[:-2]

    def __len__(self) -> int:
        """Return the number of vertices defining the convex set."""
        shape = self.shape

        if len(shape) == 0:
            msg = "len() of unsized credal set"
            raise TypeError(msg)

        return shape[0]

    def __array__(self, dtype: DTypeLike = None, copy: bool | None = None) -> np.ndarray:
        """Get the underlying numpy array of vertices."""
        if dtype is None and not copy:
            return self.tensor.numpy()

        return np.asarray(self.tensor, dtype=dtype, copy=copy)

    def lower(self) -> torch.Tensor:
        """Compute the lower envelope of the convex credal set.

        For a convex hull, the lower envelope is the element-wise minimum of its vertices.
        """
        return torch.min(self.tensor, dim=-2)[0]

    def upper(self) -> torch.Tensor:
        """Compute the upper envelope of the convex credal set.

        For a convex hull, the upper envelope is the element-wise maximum of its vertices.
        """
        return torch.max(self.tensor, dim=-2)[0]

    def copy(self) -> Self:
        """Create a copy of the credal set."""
        return type(self)(tensor=self.tensor.clone())

    def to(self, device: str | torch.device) -> Self:
        """Move the underlying array to the specified device."""
        if device == self.device:
            return self

        return type(self)(tensor=self.tensor.to(device))

    def __eq__(self, value: Any) -> Self:  # ty: ignore[invalid-method-override]  # noqa: ANN401, PYI032
        """Vectorized equality comparison."""
        return torch.equal(self.tensor, value)  # ty: ignore[invalid-return-type]

    def __hash__(self) -> int:
        """Compute the hash of the credal set."""
        return super().__hash__()


@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchProbabilityIntervalsCredalSet(TorchCategoricalCredalSet, ProbabilityIntervalsCredalSet[torch.Tensor]):
    """A credal set defined by probability intervals over outcomes.

    This represents uncertainty through lower and upper probability bounds for each class.
    Each bound is stored as a seperate numpy array of shape (..., num_classes).
    """

    lower_bounds: torch.Tensor
    upper_bounds: torch.Tensor

    @override
    @classmethod
    def from_data(cls, data: torch.Tensor, distribution_axis: int = -1) -> Self:
        """Create probability intervals from a stacked tensor.

        Args:
            data: Tensor of shape ``(..., 2, num_classes)`` where ``data[..., 0, :]``
                contains lower bounds and ``data[..., 1, :]`` contains upper bounds.
                The bounds axis is selected by ``distribution_axis``.
            distribution_axis: Axis that holds the ``[lower, upper]`` pair.
                Defaults to the second-to-last axis (``-2`` with implicit class axis at
                ``-1``).

        Returns:
            A new :class:`TorchProbabilityIntervalsCredalSet` instance.

        """
        # Normalise negative axis before moving
        if distribution_axis < 0:
            distribution_axis += data.ndim

        # Bring the bounds axis to position -2 so data has shape (..., 2, num_classes)
        data = torch.moveaxis(data, distribution_axis, -2)

        lower_bounds = data[..., 0, :]
        upper_bounds = data[..., 1, :]

        return cls(lower_bounds=lower_bounds, upper_bounds=upper_bounds)

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

    def to_torch(self, dtype: torch.dtype | None = None, copy: bool | None = None) -> torch.Tensor:
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

    def __array__(self, dtype: np.dtype | None = None) -> torch.Tensor:
        """Get the intervals as a stacked array with shape (..., 2, num_classes).

        Args:
            dtype: Desired data type.
            copy: Whether to return a copy.

        Returns:
            Stacked array of [lower_bounds, upper_bounds].
        """
        stacked = np.stack([self.lower_bounds.numpy(), self.upper_bounds.numpy()], axis=-2)

        if dtype is None:
            return stacked

        return stacked.as_type(dtype)

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

    def __eq__(self, value: Any) -> Self:  # ty: ignore[invalid-method-override]  # noqa: ANN401, PYI032
        """Vectorized equality comparison."""
        return torch.equal(self, value)  # ty: ignore[invalid-return-type, invalid-argument-type]

    def __hash__(self) -> int:
        """Compute the hash of the intervals."""
        return super().__hash__()


create_probability_intervals.register(torch.Tensor, TorchProbabilityIntervalsCredalSet.from_sample)
create_convex_credal_set.register(torch.Tensor, TorchConvexCredalSet.from_data)
