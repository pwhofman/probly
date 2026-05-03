"""JAX-based Gaussian distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, override

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats

from probly.representation._protected_axis.jax import JaxAxisProtected
from probly.representation.distribution._common import (
    GaussianDistribution,
    GaussianDistributionSample,
    create_gaussian_distribution,
)
from probly.representation.sample.jax import JaxArraySample

if TYPE_CHECKING:
    from collections.abc import Iterator


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
    def _rotate_sample_axis(self, source_axis: int, dest_axis: int) -> JaxGaussianDistribution:
        """Rotate the batch sample axis of ``mean`` and ``var``.

        Both fields share the same shape (no protected trailing axes), so the
        rotation is a straight ``jnp.moveaxis`` on each.
        """
        if source_axis == dest_axis:
            return self
        mean = jnp.moveaxis(self.mean, source_axis, dest_axis)
        var = jnp.moveaxis(self.var, source_axis, dest_axis)
        return JaxGaussianDistribution(mean=mean, var=var)

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


class JaxGaussianDistributionSample(  # ty:ignore[conflicting-metaclass]
    GaussianDistributionSample,  # intentionally parameterless: T propagates as unbound
    JaxArraySample,
):
    """Sample of Gaussian distributions stored as JAX arrays.

    Note: this class intentionally does not parametrize
    ``GaussianDistributionSample`` because :class:`JaxArraySample` is
    non-generic and binds the ``Sample`` type parameter to ``jax.Array``;
    the unbound type variable from the parent class is left unbound rather
    than dropped from the MRO. See the corresponding note on
    :class:`JaxCategoricalDistributionSample` for the same rationale.

    Rotation of ``sample_axis`` is handled by :meth:`JaxArraySample.samples`,
    which delegates to ``JaxGaussianDistribution._rotate_sample_axis``.
    """

    sample_space: ClassVar[type[GaussianDistribution]] = JaxGaussianDistribution

    @override
    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        return super().__instancehook__(instance)


@create_gaussian_distribution.register(jax.Array)
def _create_jax_gaussian_distribution_from_array(
    mean: jax.Array, var: jax.Array | None = None
) -> JaxGaussianDistribution:
    """Create a JaxGaussianDistribution from JAX arrays."""
    if var is None:
        if mean.shape[-1] != 2:
            msg = "If var is not provided, mean must have shape (..., 2) where the last axis contains [mean, var]"
            raise ValueError(msg)
        return JaxGaussianDistribution(mean=mean[..., 0], var=mean[..., 1])
    return JaxGaussianDistribution(mean=mean, var=var)
