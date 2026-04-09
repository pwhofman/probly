"""Torch-backed categorical credal set representations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, override

import torch

from probly.representation._protected_axis.torch import TorchAxisProtected
from probly.representation.credal_set._common import (
    CategoricalCredalSet,
    ConvexCredalSet,
    ProbabilityIntervalsCredalSet,
    create_convex_credal_set,
    create_probability_intervals,
)
from probly.representation.distribution.torch_categorical import TorchTensorCategoricalDistribution
from probly.representation.sample.torch import TorchTensorSample

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from probly.representation.sample._common import Sample


def _ensure_torch_categorical_distribution(value: object) -> TorchTensorCategoricalDistribution:
    if isinstance(value, TorchTensorCategoricalDistribution):
        return value
    return TorchTensorCategoricalDistribution(probabilities=torch.as_tensor(value))


def _sample_probabilities(
    sample: TorchTensorSample[TorchTensorCategoricalDistribution],
    distribution_axis: int = -1,
) -> torch.Tensor:
    sample_values = sample.samples
    if not isinstance(sample_values, TorchTensorCategoricalDistribution):
        msg = "Torch categorical credal sets require samples of TorchTensorCategoricalDistribution."
        raise TypeError(msg)

    if distribution_axis != -1:
        msg = "distribution_axis is only supported as -1 for distribution-backed samples."
        raise ValueError(msg)

    return sample_values.probabilities


class TorchCategoricalCredalSet(CategoricalCredalSet, ABC):
    """Base class for torch-backed categorical credal sets."""

    @override
    @classmethod
    def from_sample(cls, sample: Sample[TorchTensorCategoricalDistribution]) -> Self:
        torch_sample = TorchTensorSample.from_iterable(sample.samples, sample_dim=0)
        if not isinstance(torch_sample.tensor, TorchTensorCategoricalDistribution):
            msg = "Expected TorchTensorSample[TorchTensorCategoricalDistribution] for categorical credal sets."
            raise TypeError(msg)
        return cls.from_torch_sample(cast("TorchTensorSample[TorchTensorCategoricalDistribution]", torch_sample))

    @classmethod
    @abstractmethod
    def from_torch_sample(
        cls,
        sample: TorchTensorSample[TorchTensorCategoricalDistribution],
        distribution_axis: int = -1,
    ) -> Self:
        """Create a credal set from categorical distribution samples."""


@dataclass(frozen=True, slots=True, weakref_slot=True)  # ty:ignore[conflicting-metaclass]
class TorchConvexCredalSet(
    TorchAxisProtected[Any],
    TorchCategoricalCredalSet,
    ConvexCredalSet,
):
    """A convex hull over a finite set of categorical distributions."""

    tensor: TorchTensorCategoricalDistribution
    protected_axes: ClassVar[dict[str, int]] = {"tensor": 1}

    def __post_init__(self) -> None:
        """Validate that the tensor contains valid categorical distributions."""
        object.__setattr__(self, "tensor", _ensure_torch_categorical_distribution(self.tensor))

    @override
    @classmethod
    def from_torch_sample(
        cls,
        sample: TorchTensorSample[TorchTensorCategoricalDistribution],
        distribution_axis: int = -1,
    ) -> Self:
        probabilities = _sample_probabilities(sample, distribution_axis)
        vertices = torch.moveaxis(probabilities, 0, -2)
        return cls(tensor=TorchTensorCategoricalDistribution(probabilities=vertices))

    @override
    @property
    def num_classes(self) -> int:
        """Get the number of classes."""
        return self.tensor.num_classes


@dataclass(frozen=True, slots=True, weakref_slot=True)  # ty:ignore[conflicting-metaclass]
class TorchProbabilityIntervalsCredalSet(
    TorchAxisProtected[Any],
    TorchCategoricalCredalSet,
    ProbabilityIntervalsCredalSet,
):
    """Credal set represented by lower/upper categorical bounds."""

    lower_bounds: torch.Tensor
    upper_bounds: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"lower_bounds": 1, "upper_bounds": 1}

    def __post_init__(self) -> None:
        """Validate that lower and upper bounds have the same shape and are valid distributions."""
        object.__setattr__(self, "lower_bounds", _ensure_torch_categorical_distribution(self.lower_bounds))
        object.__setattr__(self, "upper_bounds", _ensure_torch_categorical_distribution(self.upper_bounds))

    @override
    @classmethod
    def from_torch_sample(
        cls,
        sample: TorchTensorSample[TorchTensorCategoricalDistribution],
        distribution_axis: int = -1,
    ) -> Self:
        probabilities = _sample_probabilities(sample, distribution_axis)
        lower_bounds = torch.min(probabilities, dim=0).values
        upper_bounds = torch.max(probabilities, dim=0).values
        return cls(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

    @override
    @property
    def num_classes(self) -> int:
        """Get the number of classes."""
        return self.lower_bounds.shape[-1]

    @override
    def numpy(self, *, force: bool = False) -> NDArray[Any]:
        stacked = torch.stack([self.lower_bounds, self.upper_bounds], dim=-2)
        array = stacked.numpy(force=True)
        if force:
            return array.copy()
        return array

    def width(self) -> torch.Tensor:
        """Compute interval width for each class."""
        return self.upper_bounds - self.lower_bounds

    def contains(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Check whether probabilities are inside the intervals."""
        within_bounds = (probabilities >= self.lower_bounds) & (probabilities <= self.upper_bounds)
        return torch.all(within_bounds, dim=-1)


create_probability_intervals.register(
    TorchTensorCategoricalDistribution, TorchProbabilityIntervalsCredalSet.from_sample
)
create_convex_credal_set.register(TorchTensorSample, TorchConvexCredalSet.from_torch_sample)
