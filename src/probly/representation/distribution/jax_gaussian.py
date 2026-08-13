"""Jax-based Gaussian distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, override

import jax
from jax import numpy as jnp
import numpy as np
from scipy.stats import norm

from probly.representation._protected_axis.jax import JaxAxisProtected
from probly.representation.distribution._common import (
    GaussianDistribution,
    GaussianDistributionSample,
    create_gaussian_distribution,
)
from probly.representation.jax_functions import jax_stack
from probly.representation.sample.jax import JaxArraySample

if TYPE_CHECKING:
    from jax._src.typing import ArrayLike
    from numpy.typing import DTypeLike


@dataclass(frozen=True, slots=True, weakref_slot=True)
class JaxGaussianDistribution(JaxAxisProtected[jax.Array], GaussianDistribution[jax.Array]):
    """Gaussian distribution with array parameters."""

    mean: jax.Array
    var: jax.Array

    type: Literal["gaussian"] = "gaussian"
    protected_axes: ClassVar[dict[str, int]] = {"mean": 0, "var": 0}

    def __post_init__(self) -> None:
        """Validate shapes and variances."""
        mean = jnp.asarray(self.mean, dtype=float)
        var = jnp.asarray(self.var, dtype=float)

        if mean.shape != var.shape:
            msg = f"mean and var must have same shape, got {mean.shape} and {var.shape}."
            raise ValueError(msg)
        if jnp.any(var <= 0):
            msg = "Variance must be positive."
            raise ValueError(msg)

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "var", var)

    @property
    def std(self) -> jax.Array:
        """Get the standard deviation."""
        return jnp.sqrt(self.var)

    def quantile(self, q: float | list[float] | jax.Array) -> jax.Array:
        """Calculate the quantile function at the given points."""
        q_arr = jnp.asarray(q)
        res = norm.ppf(q_arr, loc=self.mean[..., None], scale=self.std[..., None])

        if q_arr.ndim == 0:
            return res.squeeze(-1)
        return res

    @override
    def sample(
        self,
        num_samples: int = 1,
        prng_key: ArrayLike | None = None,
    ) -> JaxArraySample[jax.Array]:
        """Draw samples and wrap them in an JaxArraySample (sample_axis=0)."""
        if prng_key is None:
            prng_key = jax.random.key(0)

        std = self.std
        z = jax.random.normal(prng_key, shape=(num_samples, *self.mean.shape))
        samples = self.mean + std * z
        return JaxArraySample(array=samples, sample_axis=0)

    def __array__(
        self,
        dtype: DTypeLike | None = None,
        /,
        *,
        copy: bool | None = None,
    ) -> np.ndarray:
        """Represent the distribution as stacked [mean, var] on the last axis."""
        stacked = jax_stack([self.mean, self.var], axis=-1)
        return np.asarray(stacked, dtype=dtype, copy=copy)

    @override
    def __eq__(self, other: Any) -> jax.Array:  # ty: ignore[invalid-method-override] # noqa: PYI032
        """Compare two Gaussians by their parameter."""
        if not isinstance(other, JaxGaussianDistribution):
            return NotImplemented
        return jnp.equal(self.mean, other.mean) & jnp.equal(self.var, other.var)

    def __hash__(self) -> int:
        """Return an identity-based hash.

        We intentionally byass ``super()`` here because protocol-heavy MROs can
        produce invalid ``super(type, obj)`` bindings at runtime. ``object``'s
        hash gives per-instance identity semantics.
        """
        return object.__hash__(self)


@create_gaussian_distribution.register(jax.Array)
def _(mean: jax.Array, var: jax.Array | None = None) -> JaxGaussianDistribution:
    """Create a JaxGaussianDistribution from jax arrays."""
    if var is None:
        if mean.shape[-1] != 2:
            msg = "If var is not provided, mean must have shape (..., 2) where the last axis contains [mean, var]."
            raise ValueError(msg)
        return JaxGaussianDistribution(mean=mean[..., 0], var=mean[..., 1])
    return JaxGaussianDistribution(mean=mean, var=var)


class JaxGaussianDistributionSample(  # ty:ignore[conflicting-metaclass]
    GaussianDistributionSample[JaxGaussianDistribution],
    JaxArraySample[JaxGaussianDistribution],
):
    """Sample type for empirical second-order Gaussian distributions."""

    sample_space: ClassVar[type[GaussianDistribution]] = JaxGaussianDistribution

    @override
    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        return super().__instancehook__(instance)
