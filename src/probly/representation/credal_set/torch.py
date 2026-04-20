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
    create_probability_intervals_from_bounds,
    create_probability_intervals_from_lower_upper_array,
)
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.representation.sample.torch import TorchSample

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from probly.representation.sample._common import Sample


def _ensure_torch_categorical_distribution(value: object) -> TorchCategoricalDistribution:
    if isinstance(value, TorchCategoricalDistribution):
        return value
    return TorchCategoricalDistribution(torch.as_tensor(value))


def _sample_probabilities(
    sample: TorchSample[TorchCategoricalDistribution],
) -> torch.Tensor:
    sample_values = sample.samples
    if not isinstance(sample_values, TorchCategoricalDistribution):
        msg = "Torch categorical credal sets require samples of TorchCategoricalDistribution."
        raise TypeError(msg)

    return sample_values.unnormalized_probabilities


class TorchCategoricalCredalSet(CategoricalCredalSet, ABC):
    """Base class for torch-backed categorical credal sets."""

    @override
    @classmethod
    def from_sample(cls, sample: Sample[TorchCategoricalDistribution]) -> Self:
        torch_sample = TorchSample.from_iterable(sample.samples, sample_dim=0)
        if not isinstance(torch_sample.tensor, TorchCategoricalDistribution):
            msg = "Expected TorchSample[TorchCategoricalDistribution] for categorical credal sets."
            raise TypeError(msg)
        return cls.from_torch_sample(cast("TorchSample[TorchCategoricalDistribution]", torch_sample))

    @classmethod
    @abstractmethod
    def from_torch_sample(
        cls,
        sample: TorchSample[TorchCategoricalDistribution],
    ) -> Self:
        """Create a credal set from categorical distribution samples."""


@dataclass(frozen=True, slots=True, weakref_slot=True)  # ty:ignore[conflicting-metaclass]
class TorchConvexCredalSet(
    TorchAxisProtected[Any],
    TorchCategoricalCredalSet,
    ConvexCredalSet,
):
    """A convex hull over a finite set of categorical distributions."""

    tensor: TorchCategoricalDistribution
    protected_axes: ClassVar[dict[str, int]] = {"tensor": 1}

    def __post_init__(self) -> None:
        """Validate that the tensor contains valid categorical distributions."""
        object.__setattr__(self, "tensor", _ensure_torch_categorical_distribution(self.tensor))

    @override
    @classmethod
    def from_torch_sample(
        cls,
        sample: TorchSample[TorchCategoricalDistribution],
    ) -> Self:
        probabilities = _sample_probabilities(sample)
        vertices = torch.moveaxis(probabilities, 0, -2)
        return cls(tensor=TorchCategoricalDistribution(vertices))

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

    @override
    @classmethod
    def from_torch_sample(
        cls,
        sample: TorchSample[TorchCategoricalDistribution],
    ) -> Self:
        probabilities = _sample_probabilities(sample)
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


create_probability_intervals.register(TorchCategoricalDistribution, TorchProbabilityIntervalsCredalSet.from_sample)
create_probability_intervals.register(TorchSample, TorchProbabilityIntervalsCredalSet.from_torch_sample)
create_convex_credal_set.register(TorchSample, TorchConvexCredalSet.from_torch_sample)


@create_probability_intervals_from_lower_upper_array.register(torch.Tensor)
def _create_probability_intervals_from_lower_upper_array(
    bounds: torch.Tensor,
) -> TorchProbabilityIntervalsCredalSet:
    lower_bounds, upper_bounds = bounds.reshape(*bounds.shape[:-1], 2, -1).unbind(dim=-2)
    return TorchProbabilityIntervalsCredalSet(lower_bounds, upper_bounds)


@create_probability_intervals_from_bounds.register(torch.Tensor)
def _create_probability_intervals_from_bounds(
    probs: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> TorchProbabilityIntervalsCredalSet:
    return TorchProbabilityIntervalsCredalSet(probs - lower, probs + upper)
