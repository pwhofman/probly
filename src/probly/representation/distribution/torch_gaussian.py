"""Torch-based Gaussian distribution representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, override

from ndlike.protected_axis.torch import TorchAxisProtected
import torch

from probly.representation.distribution._common import (
    GaussianDistribution,
    GaussianDistributionSample,
    create_gaussian_distribution,
)
from probly.representation.sample.torch import TorchSample

if TYPE_CHECKING:
    import builtins
    from collections.abc import Callable

    from probly.representation.representation import Representation


@create_gaussian_distribution.register(torch.Tensor)
@dataclass(frozen=True, slots=True, weakref_slot=True)
class TorchGaussianDistribution(
    TorchAxisProtected["TorchGaussianDistribution", "TorchGaussianDistribution | torch.Tensor | float"],
    GaussianDistribution[torch.Tensor],
):
    """Gaussian distribution with torch tensor parameters.

    Addition and subtraction combine independent Gaussian variances. Constant
    shifts and sign changes preserve variance. Arithmetic requires ``alpha=1``
    and does not support ``out``; use the returned distribution instead.
    """

    mean: torch.Tensor = field()
    var: torch.Tensor = field()

    type: ClassVar[Literal["gaussian"]] = "gaussian"

    permitted_functions: ClassVar[set[Callable]] = {torch.add, torch.sub, torch.neg, torch.positive, torch.eq, torch.ne}

    @override
    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[builtins.type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        options = {} if kwargs is None else kwargs
        if func in (torch.add, torch.sub, torch.neg, torch.positive):
            # ndlike does not run the variance correction for explicit outputs,
            # and its correction hook does not receive keyword arguments.
            if options.get("out") is not None:
                msg = "Gaussian arithmetic does not support out; use the returned distribution."
                raise TypeError(msg)
            if options.get("alpha", 1) != 1:
                msg = "Gaussian addition and subtraction require alpha=1."
                raise ValueError(msg)
        return super(TorchGaussianDistribution, cls).__torch_function__(func, types, args, kwargs)

    @override
    def _postprocess_elementwise_result(
        self, values: dict[str, Any], *, func: Callable, operands: tuple[object, ...]
    ) -> dict[str, Any]:
        """Preserve variance under shifts and sum variances of independent Gaussians."""
        if func in (torch.add, torch.sub):
            left, right = operands
            if isinstance(left, TorchGaussianDistribution) and isinstance(right, TorchGaussianDistribution):
                values["var"] = left.var + right.var
                return values
        values["var"] = torch.broadcast_to(self.var, values["mean"].shape)
        return values

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
    def __eq__(self, other: Any) -> torch.Tensor:  # noqa: PYI032
        """Compare two Gaussians by their parameters."""
        if not isinstance(other, TorchGaussianDistribution):
            return NotImplemented
        return torch.eq(self, other)  # ty: ignore[no-matching-overload]

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

    sample_space: ClassVar[type[Representation]] = TorchGaussianDistribution

    @override
    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        return super().__instancehook__(instance)
