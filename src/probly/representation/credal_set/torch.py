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
    DistanceBasedCredalSet,
    ProbabilityIntervalsCredalSet,
    create_convex_credal_set,
    create_distance_based_credal_set,
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
class TorchDistanceBasedCredalSet(
    TorchAxisProtected[Any],
    TorchCategoricalCredalSet,
    DistanceBasedCredalSet,
):
    """Distance-based credal set around a nominal categorical distribution."""

    nominal: TorchCategoricalDistribution
    radius: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"nominal": 0, "radius": 0}

    def __post_init__(self) -> None:
        """Validate that nominal is a valid categorical distribution and radius is non-negative."""
        object.__setattr__(self, "nominal", _ensure_torch_categorical_distribution(self.nominal))
        object.__setattr__(self, "radius", torch.as_tensor(self.radius))

    @override
    @classmethod
    def from_torch_sample(
        cls,
        sample: TorchSample[TorchCategoricalDistribution],
    ) -> Self:
        probabilities = _sample_probabilities(sample)
        nominal = torch.mean(probabilities, dim=0)
        diff = torch.abs(probabilities - nominal)
        tv_dists = 0.5 * torch.sum(diff, dim=-1)
        radius = torch.max(tv_dists, dim=0).values
        return cls(
            nominal=TorchCategoricalDistribution(nominal),
            radius=torch.as_tensor(radius),
        )

    @override
    @property
    def num_classes(self) -> int:
        """Return the number of classes in the credal set."""
        return self.nominal.num_classes

    def lower(self) -> torch.Tensor:
        """Compute the lower envelope of the credal set.

        For L1/TV distance, the tightest element-wise lower bound is max(0, nominal - radius).
        """
        nominal = self.nominal.unnormalized_probabilities
        r = self.radius
        if isinstance(r, torch.Tensor) and r.dim() == nominal.dim() - 1:
            r = r.unsqueeze(-1)

        return torch.clamp(nominal - r, min=0.0, max=1.0)

    def upper(self) -> torch.Tensor:
        """Compute the upper envelope of the credal set.

        For L1/TV distance, the tightest element-wise upper bound is min(1, nominal + radius).
        """
        nominal = self.nominal.unnormalized_probabilities
        r = self.radius
        if isinstance(r, torch.Tensor) and r.dim() == nominal.dim() - 1:
            r = r.unsqueeze(-1)

        return torch.clamp(nominal + r, min=0.0, max=1.0)


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
create_distance_based_credal_set.register(TorchSample, TorchDistanceBasedCredalSet.from_torch_sample)


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
