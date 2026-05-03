"""Torch-based Gaussian distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, override

import torch

from probly.representation._protected_axis.torch import TorchAxisProtected
from probly.representation.distribution._common import (
    GaussianDistribution,
    GaussianDistributionSample,
    create_gaussian_distribution,
)
from probly.representation.sample.torch import TorchSample

if TYPE_CHECKING:
    from collections.abc import Iterator


@create_gaussian_distribution.register(torch.Tensor)
@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchGaussianDistribution(TorchAxisProtected[Any], GaussianDistribution[torch.Tensor]):
    """Gaussian distribution with torch tensor parameters."""

    mean: torch.Tensor
    var: torch.Tensor

    type: Literal["gaussian"] = "gaussian"
    protected_axes: ClassVar[dict[str, int]] = {"mean": 0, "var": 0}

    def __post_init__(self) -> None:
        """Validate shapes and variances."""
        if not isinstance(self.mean, torch.Tensor):
            msg = "mean must be a torch tensor."
            raise TypeError(msg)
        if not isinstance(self.var, torch.Tensor):
            msg = "var must be a torch tensor."
            raise TypeError(msg)

        if self.mean.shape != self.var.shape:
            msg = f"mean and var must have same shape, got {self.mean.shape} and {self.var.shape}"
            raise ValueError(msg)
        if torch.any(self.var <= 0):
            msg = "Variance must be positive"
            raise ValueError(msg)

    @property
    def std(self) -> torch.Tensor:
        """Get the standard deviation."""
        return torch.sqrt(self.var)

    def quantile(self, q: float | list[float] | torch.Tensor) -> torch.Tensor:
        """Calculate the quantile function at the given points."""
        q_tensor = torch.as_tensor(q, dtype=self.mean.dtype, device=self.mean.device)
        normal = torch.distributions.Normal(self.mean[..., None], self.std[..., None])
        res = normal.icdf(q_tensor)

        if q_tensor.ndim == 0:
            return res.squeeze(-1)
        return res

    @override
    def sample(
        self,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
    ) -> TorchSample[torch.Tensor]:
        """Draw samples and wrap them in a TorchSample (sample_dim=0)."""
        std = self.std
        expanded_mean = self.mean.expand(num_samples, *self.mean.shape)
        expanded_std = std.expand(num_samples, *std.shape)
        samples = torch.normal(mean=expanded_mean, std=expanded_std, generator=rng)
        return TorchSample(tensor=samples, sample_dim=0)

    @override
    def __iter__(self) -> Iterator[Any]:
        return self.mean.__iter__()

    @override
    def __eq__(self, other: Any) -> torch.Tensor:  # ty: ignore[invalid-method-override]  # noqa: PYI032
        """Compare two Gaussians by their parameters."""
        if not isinstance(other, TorchGaussianDistribution):
            return NotImplemented
        return torch.eq(self.mean, other.mean) & torch.eq(self.var, other.var)

    def __hash__(self) -> int:
        """Return an identity-based hash.

        We intentionally bypass ``super()`` here because protocol-heavy MROs can
        produce invalid ``super(type, obj)`` bindings at runtime. ``object``'s
        hash gives per-instance identity semantics.
        """
        return object.__hash__(self)


class TorchGaussianDistributionSample(  # ty:ignore[conflicting-metaclass]
    GaussianDistributionSample[TorchGaussianDistribution],
    TorchSample[TorchGaussianDistribution],
):
    """Sample type for empirical second-order Gaussian distributions."""

    sample_space: ClassVar[type[GaussianDistribution]] = TorchGaussianDistribution

    @override
    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        return super().__instancehook__(instance)
