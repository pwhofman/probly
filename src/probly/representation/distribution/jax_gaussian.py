"""JAX-based Gaussian distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, Unpack, override

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats

from probly.representation._protected_axis.jax import JaxAxisProtected
from probly.representation.distribution._common import (
    GaussianDistribution,
    GaussianDistributionSample,
    create_gaussian_distribution,
)
from probly.representation.sample._common import Sample, SampleParams
from probly.representation.sample.jax import JaxArraySample

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


@dataclass(frozen=True, slots=True, weakref_slot=True)
class JaxGaussianDistribution(JaxAxisProtected[Any], GaussianDistribution[jax.Array]):
    """Gaussian distribution with JAX array parameters."""

    mean: jax.Array
    var: jax.Array

    type: Literal["gaussian"] = "gaussian"
    protected_axes: ClassVar[dict[str, int]] = {"mean": 0, "var": 0}

    def __post_init__(self) -> None:
        """Validate shapes and variances."""
        if not isinstance(self.mean, jax.Array):
            msg = "mean must be a JAX array."
            raise TypeError(msg)
        if not isinstance(self.var, jax.Array):
            msg = "var must be a JAX array."
            raise TypeError(msg)

        if self.mean.shape != self.var.shape:
            msg = f"mean and var must have same shape, got {self.mean.shape} and {self.var.shape}"
            raise ValueError(msg)
        if jnp.any(self.var <= 0):
            msg = "Variance must be positive"
            raise ValueError(msg)

    @property
    def std(self) -> jax.Array:
        """Get the standard deviation."""
        return jnp.sqrt(self.var)

    def quantile(self, q: float | list[float] | jax.Array) -> jax.Array:
        """Calculate the quantile function at the given points."""
        q_arr = jnp.asarray(q, dtype=self.mean.dtype)
        res = jstats.norm.ppf(q_arr, loc=self.mean[..., None], scale=self.std[..., None])

        if q_arr.ndim == 0:
            return jnp.squeeze(res, axis=-1)
        return res

    @override
    def sample(
        self,
        num_samples: int = 1,
        rng: jax.Array | None = None,
    ) -> JaxArraySample:
        """Draw samples and wrap them in a JaxArraySample (sample_axis=0).

        Args:
            num_samples: Number of samples to draw.
            rng: A ``jax.Array`` random key. JAX has no global RNG, so when ``None`` is
                passed a deterministic default key is used.

        Returns:
            A ``JaxArraySample`` of shape ``(num_samples, *mean.shape)`` with ``sample_axis=0``.
        """
        key = jax.random.key(0) if rng is None else rng

        std = self.std
        standard = jax.random.normal(key, shape=(num_samples, *self.mean.shape), dtype=self.mean.dtype)
        samples = self.mean + std * standard
        return JaxArraySample(array=samples, sample_axis=0)

    @override
    def __iter__(self) -> Iterator[Any]:
        return self.mean.__iter__()

    @override
    def __eq__(self, other: Any) -> jax.Array:  # ty: ignore[invalid-method-override]  # noqa: PYI032
        """Compare two Gaussians by their parameters."""
        if not isinstance(other, JaxGaussianDistribution):
            return NotImplemented
        return jnp.equal(self.mean, other.mean) & jnp.equal(self.var, other.var)

    def __hash__(self) -> int:
        """Return an identity-based hash.

        We intentionally bypass ``super()`` here because protocol-heavy MROs can
        produce invalid ``super(type, obj)`` bindings at runtime. ``object``'s
        hash gives per-instance identity semantics.
        """
        return object.__hash__(self)


@dataclass(frozen=True)  # ty:ignore[conflicting-metaclass]
class JaxGaussianDistributionSample(
    GaussianDistributionSample[JaxGaussianDistribution],
    Sample[JaxGaussianDistribution],
):
    """Sample type for empirical second-order Gaussian distributions."""

    array: JaxGaussianDistribution
    sample_axis: int
    sample_space: ClassVar[type[GaussianDistribution]] = JaxGaussianDistribution

    @override
    @classmethod
    def from_iterable(
        cls,
        samples: Iterable[JaxGaussianDistribution],
        weights: Iterable[float] | None = None,
        **_kwargs: Unpack[SampleParams],
    ) -> Self:
        """Create a JaxGaussianDistributionSample from an iterable of distributions."""
        del weights
        sample_list = list(samples)
        if not sample_list:
            msg = "Cannot construct JaxGaussianDistributionSample from an empty iterable."
            raise ValueError(msg)
        means = jnp.stack([s.mean for s in sample_list], axis=0)
        variances = jnp.stack([s.var for s in sample_list], axis=0)
        return cls(array=JaxGaussianDistribution(mean=means, var=variances), sample_axis=0)

    @property
    @override
    def samples(self) -> Iterable[JaxGaussianDistribution]:
        """Iterate over per-sample Gaussian distributions along ``sample_axis``."""
        sample_axis = self.sample_axis
        means = self.array.mean
        variances = self.array.var
        if sample_axis != 0:
            means = jnp.moveaxis(means, sample_axis, 0)
            variances = jnp.moveaxis(variances, sample_axis, 0)
        return [JaxGaussianDistribution(mean=means[i], var=variances[i]) for i in range(means.shape[0])]

    @property
    @override
    def weights(self) -> Iterable[float] | None:
        """Return the (optional) sample weights."""
        return None

    @property
    def sample_size(self) -> int:
        """Return the number of samples."""
        return self.array.mean.shape[self.sample_axis]

    @override
    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        return super().__instancehook__(instance)


@create_gaussian_distribution.register(jax.Array)
def _(mean: jax.Array, var: jax.Array | None = None) -> JaxGaussianDistribution:
    """Create a JaxGaussianDistribution from JAX arrays."""
    if var is None:
        if mean.shape[-1] != 2:
            msg = "If var is not provided, mean must have shape (..., 2) where the last axis contains [mean, var]"
            raise ValueError(msg)
        return JaxGaussianDistribution(mean=mean[..., 0], var=mean[..., 1])
    return JaxGaussianDistribution(mean=mean, var=var)
