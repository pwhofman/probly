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
    DirichletLevelSetCredalSet,
    DistanceBasedCredalSet,
    ProbabilityIntervalsCredalSet,
    create_convex_credal_set,
    create_dirichlet_level_set_credal_set,
    create_distance_based_credal_set,
    create_distance_based_credal_set_from_center_and_radius,
    create_probability_intervals,
    create_probability_intervals_from_bounds,
    create_probability_intervals_from_lower_upper_array,
)
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.representation.sample.torch import TorchSample
from probly.utils.torch import intersection_probability

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

    def lower(self) -> torch.Tensor:
        """Return the per-class lower probability envelope of the convex hull."""
        return torch.amin(self.tensor.probabilities, dim=-2)

    def upper(self) -> torch.Tensor:
        """Return the per-class upper probability envelope of the convex hull."""
        return torch.amax(self.tensor.probabilities, dim=-2)

    @override
    @property
    def barycenter(self) -> TorchCategoricalDistribution:
        """Compute the barycenter of the convex credal set as the mean of the vertices."""
        return torch.mean(self.tensor, dim=-1)  # ty:ignore[no-matching-overload]


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

    @override
    @property
    def barycenter(self) -> TorchCategoricalDistribution:
        """Return the nominal distribution as the barycenter of the credal set."""
        return self.nominal


_LEVEL_SET_SAMPLES = 10_000


@dataclass(frozen=True, slots=True, weakref_slot=True)  # ty:ignore[conflicting-metaclass]
class TorchDirichletLevelSetCredalSet(
    TorchAxisProtected[Any],
    TorchCategoricalCredalSet,
    DirichletLevelSetCredalSet,
):
    """Dirichlet density level set credal set.

    Contains all distributions whose Dirichlet likelihood is at least
    ``threshold`` times the peak density.
    """

    alphas: torch.Tensor
    threshold: torch.Tensor
    protected_axes: ClassVar[dict[str, int]] = {"alphas": 1, "threshold": 0}

    def __post_init__(self) -> None:
        """Validate alpha parameters and threshold."""
        object.__setattr__(self, "alphas", torch.as_tensor(self.alphas))
        object.__setattr__(self, "threshold", torch.as_tensor(self.threshold))

    @override
    @classmethod
    def from_torch_sample(
        cls,
        sample: TorchSample[TorchCategoricalDistribution],
    ) -> Self:
        """Create from a torch sample (not supported).

        Raises:
            NotImplementedError: Always, as this credal set type cannot be created from samples.
        """
        msg = "DirichletLevelSetCredalSet cannot be created from samples."
        raise NotImplementedError(msg)

    @override
    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return self.alphas.shape[-1]

    def _log_relative_likelihood(self, samples: torch.Tensor) -> torch.Tensor:
        """Compute log relative likelihood of samples under this Dirichlet.

        Args:
            samples: Probability vectors, shape (n_samples, ..., K).

        Returns:
            Log relative likelihood, shape (n_samples, ...).
        """
        alpha = self.alphas  # (..., K)
        alpha_m1 = alpha - 1.0  # (..., K)

        # Log density of samples: sum((alpha_k - 1) * log(lambda_k))
        # (normalization constant cancels in ratio)
        log_density = (alpha_m1 * torch.log(samples.clamp(min=1e-30))).sum(dim=-1)

        # Log density at mode
        alpha_0 = alpha.sum(dim=-1)  # (...)
        k = alpha.shape[-1]
        mode = (alpha_m1) / (alpha_0 - k).unsqueeze(-1)  # (..., K)
        # When any alpha_k <= 1, mode doesn't exist in interior
        # Clamp mode to avoid log(0)
        mode = mode.clamp(min=1e-30)
        log_mode_density = (alpha_m1 * torch.log(mode)).sum(dim=-1)

        return log_density - log_mode_density

    def _sample_and_filter(self, n_samples: int = _LEVEL_SET_SAMPLES) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample from Dir(alpha) and filter to the level set.

        Args:
            n_samples: Number of samples to draw.

        Returns:
            Tuple of (samples, mask) where samples has shape (n_samples, ..., K)
            and mask has shape (n_samples, ...).
        """
        # Sample from Dirichlet
        dist = torch.distributions.Dirichlet(self.alphas)
        samples = dist.sample((n_samples,))  # (n_samples, ..., K)

        # Compute relative likelihood
        log_rl = self._log_relative_likelihood(samples)  # (n_samples, ...)
        rl = torch.exp(log_rl)  # (n_samples, ...)

        # Mask: keep samples in the level set
        mask = rl >= self.threshold  # (n_samples, ...)
        return samples, mask

    def lower(self) -> torch.Tensor:
        """Compute per-class lower bounds via Monte Carlo sampling.

        Returns:
            Lower probability bounds, shape (..., K).
        """
        samples, mask = self._sample_and_filter()
        # Replace non-accepted samples with inf so they don't affect amin
        masked = torch.where(mask.unsqueeze(-1), samples, torch.tensor(float("inf")))
        result = masked.amin(dim=0)
        # If no samples accepted (e.g., threshold too high), fall back to mode
        no_accepted = ~mask.any(dim=0)
        if no_accepted.any():
            alpha_0 = self.alphas.sum(dim=-1)
            k = self.alphas.shape[-1]
            mode = (self.alphas - 1.0).clamp(min=0.0) / (alpha_0 - k).unsqueeze(-1).clamp(min=1.0)
            mode = mode / mode.sum(dim=-1, keepdim=True)
            result = torch.where(no_accepted.unsqueeze(-1), mode, result)
        return result.clamp(min=0.0, max=1.0)

    def upper(self) -> torch.Tensor:
        """Compute per-class upper bounds via Monte Carlo sampling.

        Returns:
            Upper probability bounds, shape (..., K).
        """
        samples, mask = self._sample_and_filter()
        masked = torch.where(mask.unsqueeze(-1), samples, torch.tensor(float("-inf")))
        result = masked.amax(dim=0)
        no_accepted = ~mask.any(dim=0)
        if no_accepted.any():
            alpha_0 = self.alphas.sum(dim=-1)
            k = self.alphas.shape[-1]
            mode = (self.alphas - 1.0).clamp(min=0.0) / (alpha_0 - k).unsqueeze(-1).clamp(min=1.0)
            mode = mode / mode.sum(dim=-1, keepdim=True)
            result = torch.where(no_accepted.unsqueeze(-1), mode, result)
        return result.clamp(min=0.0, max=1.0)

    @override
    @property
    def barycenter(self) -> TorchCategoricalDistribution:
        """Return the Dirichlet mean as the barycenter."""
        return TorchCategoricalDistribution(self.alphas / self.alphas.sum(dim=-1, keepdim=True))


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

    def lower(self) -> torch.Tensor:
        """Return the per-class lower probability envelope (alias for ``lower_bounds``)."""
        return self.lower_bounds

    def upper(self) -> torch.Tensor:
        """Return the per-class upper probability envelope (alias for ``upper_bounds``)."""
        return self.upper_bounds

    def contains(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Check whether probabilities are inside the intervals."""
        within_bounds = (probabilities >= self.lower_bounds) & (probabilities <= self.upper_bounds)
        return torch.all(within_bounds, dim=-1)

    @override
    @property
    def barycenter(self) -> TorchCategoricalDistribution:
        return TorchCategoricalDistribution(intersection_probability(self.lower_bounds, self.upper_bounds))


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


@create_distance_based_credal_set_from_center_and_radius.register(torch.Tensor)
def _create_distance_based_credal_set_from_center_and_radius(
    center: torch.Tensor,
    radius: torch.Tensor,
) -> TorchDistanceBasedCredalSet:
    return TorchDistanceBasedCredalSet(
        nominal=TorchCategoricalDistribution(center),
        radius=radius,
    )


@create_distance_based_credal_set_from_center_and_radius.register(TorchCategoricalDistribution)
def _create_distance_based_credal_set_from_categorical_distribution(
    center: TorchCategoricalDistribution,
    radius: torch.Tensor,
) -> TorchDistanceBasedCredalSet:
    return TorchDistanceBasedCredalSet(
        nominal=center,
        radius=torch.as_tensor(radius),
    )


@create_dirichlet_level_set_credal_set.register(torch.Tensor)
def _create_dirichlet_level_set_from_tensor(
    alphas: torch.Tensor,
    threshold: torch.Tensor,
) -> TorchDirichletLevelSetCredalSet:
    return TorchDirichletLevelSetCredalSet(
        alphas=alphas,
        threshold=torch.as_tensor(threshold),
    )
