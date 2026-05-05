"""Torch-based mixture distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import Any, ClassVar, Literal, cast, override

import torch

from probly.representation._protected_axis.torch import TorchAxisProtected
from probly.representation.distribution._common import (
    Distribution,
    MixtureDistribution,
    create_dirichlet_distribution_from_alphas,
    create_dirichlet_mixture_distribution_from_alphas_and_weights,
)
from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.representation.distribution.torch_dirichlet import TorchDirichletDistribution
from probly.representation.sample.torch import TorchSample
from probly.representation.torch_functions import torch_average
from probly.representation.torch_like import TorchLike


@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchMixtureDistribution[D: Distribution, T: TorchLike](
    TorchAxisProtected[T],
    MixtureDistribution[D, T],
):
    """Mixture distribution with torch tensor mixture weights.

    The final batch axis of ``components`` and ``mixture_weights`` is the
    protected component axis. The exposed shape is the shared batch shape before
    that component axis.
    """

    components: D
    mixture_weights: torch.Tensor

    type: Literal["mixture"] = "mixture"
    protected_axes: ClassVar[dict[str, int]] = {"components": 1, "mixture_weights": 1}

    def __post_init__(self) -> None:
        """Validate mixture component and weight shapes."""
        if not isinstance(self.components, Distribution) or not isinstance(self.components, TorchAxisProtected):
            msg = "components must be a torch distribution array."
            raise TypeError(msg)
        if not isinstance(self.mixture_weights, torch.Tensor):
            msg = "mixture_weights must be a torch tensor."
            raise TypeError(msg)

        if self.mixture_weights.ndim < 1:
            msg = "mixture_weights must have at least one dimension."
            raise ValueError(msg)
        if self.mixture_weights.shape[-1] < 1:
            msg = "Mixture distribution requires at least one component."
            raise ValueError(msg)
        if not torch.all(torch.isfinite(self.mixture_weights)):
            msg = "mixture_weights must be finite."
            raise ValueError(msg)
        if torch.any(self.mixture_weights < 0):
            msg = "mixture_weights must be non-negative."
            raise ValueError(msg)
        if torch.any(torch.sum(self.mixture_weights, dim=-1) <= 0):
            msg = "mixture_weights must have positive sums along the component axis."
            raise ValueError(msg)

        try:
            component_shape = tuple(int(dim) for dim in cast("Any", self.components).shape)
        except (AttributeError, TypeError) as exc:
            msg = "components must expose a shape attribute."
            raise TypeError(msg) from exc

        if component_shape != tuple(self.mixture_weights.shape):
            msg = (
                "components shape must match mixture_weights shape, "
                f"got {component_shape} and {tuple(self.mixture_weights.shape)}."
            )
            raise ValueError(msg)

    @property
    def normalized_mixture_weights(self) -> torch.Tensor:
        """Return mixture weights normalized along the component axis."""
        return self.mixture_weights / torch.sum(self.mixture_weights, dim=-1, keepdim=True)

    @override
    def sample(
        self,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
    ) -> TorchSample[T]:
        """Draw samples from the mixture distribution.

        Args:
            num_samples: Number of samples to draw.
            rng: Optional torch random generator used for component selection
                and forwarded to components that accept an ``rng`` argument.

        Returns:
            Samples drawn from the selected mixture components.
        """
        flat_probabilities = self.normalized_mixture_weights.reshape((-1, self.mixture_weights.shape[-1]))
        flat_indices = torch.multinomial(flat_probabilities, num_samples=num_samples, replacement=True, generator=rng)

        components: TorchAxisProtected = self.components  # ty:ignore[invalid-assignment]

        flat_components = components.reshape((-1, self.mixture_weights.shape[-1]))
        gathered_components = flat_components.gather(dim=-1, index=flat_indices)
        selected_components = gathered_components.transpose(0, 1).reshape(num_samples, *self.shape)

        sample_method = selected_components.sample  # ty:ignore[unresolved-attribute]
        if "rng" in signature(sample_method).parameters:
            component_sample: TorchSample = sample_method(1, rng=rng)
        else:
            component_sample = sample_method(1)

        if not isinstance(component_sample, TorchSample):
            msg = "Torch mixture components must return a TorchSample."
            raise TypeError(msg)

        selected = component_sample.samples[0]

        return TorchSample(tensor=selected, sample_dim=0)

    @override
    def __eq__(self, value: Any) -> torch.Tensor:  # ty: ignore[invalid-method-override]  # noqa: PYI032
        """Vectorized equality comparison."""
        if not isinstance(value, TorchMixtureDistribution):
            return NotImplemented

        component_eq = self.components == value.components
        if component_eq is NotImplemented:
            return NotImplemented
        component_eq = torch.as_tensor(component_eq, device=self.mixture_weights.device)
        while component_eq.ndim > self.ndim:
            component_eq = torch.all(component_eq, dim=-1)

        weight_eq = torch.all(
            torch.eq(self.normalized_mixture_weights, value.normalized_mixture_weights),
            dim=-1,
        )
        return component_eq & weight_eq

    def __hash__(self) -> int:
        """Return an identity-based hash."""
        return object.__hash__(self)

    @property
    def mean(self) -> T:
        """Compute the mean of the mixture distribution."""
        component_means = self.components.mean  # ty:ignore[unresolved-attribute]
        weights = self.mixture_weights

        return torch_average(component_means, dim=-1, weights=weights)  # ty:ignore[invalid-return-type]


class TorchDirichletMixtureDistribution(  # ty:ignore[conflicting-metaclass]
    TorchMixtureDistribution[TorchDirichletDistribution, TorchCategoricalDistribution],
    MixtureDistribution[TorchDirichletDistribution, TorchCategoricalDistribution],
):
    """Mixture distribution with Dirichlet components and torch tensor mixture weights."""

    component_type: ClassVar[type[TorchDirichletDistribution]] = TorchDirichletDistribution
    components: TorchDirichletDistribution
    mixture_weights: torch.Tensor

    @override
    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        return super().__instancehook__(instance)


@create_dirichlet_mixture_distribution_from_alphas_and_weights.register(torch.Tensor)
def _(
    alphas: torch.Tensor,
    weights: torch.Tensor,
) -> TorchMixtureDistribution[TorchDirichletDistribution, TorchCategoricalDistribution]:
    """Create a TorchMixtureDistribution from alphas and weights."""
    return TorchMixtureDistribution(
        components=cast("TorchDirichletDistribution", create_dirichlet_distribution_from_alphas(alphas)),
        mixture_weights=weights,
    )
