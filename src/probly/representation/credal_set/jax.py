"""Jax-backed categorical credal set representations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, override
import weakref

import jax
import jax.numpy as jnp

from probly.representation._protected_axis.jax import JaxAxisProtected
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
from probly.representation.distribution.jax_categorical import (
    JaxCategoricalDistribution,
    JaxProbabilityCategoricalDistribution,
)
from probly.representation.jax_functions import jax_mean
from probly.representation.sample.jax import JaxArraySample
from probly.utils.jax import intersection_probability

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from probly.representation.sample._common import Sample


def _ensure_jax_categorical_distribution(value: object) -> JaxCategoricalDistribution:
    if isinstance(value, JaxCategoricalDistribution):
        return value
    return JaxProbabilityCategoricalDistribution(jnp.asarray(value))


def _sample_probabilities(
    sample: JaxArraySample[JaxCategoricalDistribution],
) -> jax.Array:
    sample_values = sample.samples
    if not isinstance(sample_values, JaxCategoricalDistribution):
        msg = "Jax categorical credal sets require samples of JaxCategoricalDistribution."
        raise TypeError(msg)

    return sample_values.probabilities


class JaxCategoricalCredalSet(CategoricalCredalSet, ABC):
    """Base class for jax-backed categorical credal sets."""

    @override
    @classmethod
    def from_sample(cls, sample: Sample[JaxCategoricalDistribution]) -> Self:
        jax_sample = JaxArraySample.from_iterable(sample.samples, sample_axis=0)
        if not isinstance(jax_sample.array, JaxCategoricalDistribution):
            msg = "Expected JaxArraySample[JaxCategoricalDistribution] for categorical credal sets."
            raise TypeError(msg)
        return cls.from_jax_sample(cast("JaxArraySample[JaxCategoricalDistribution]", jax_sample))

    @classmethod
    @abstractmethod
    def from_jax_sample(
        cls,
        sample: JaxArraySample[JaxCategoricalDistribution],
    ) -> Self:
        """Create a credal set from categorical distribution samples."""


@dataclass(frozen=True, slots=True, weakref_slot=True)  # ty:ignore[conflicting-metaclass]
class JaxConvexCredalSet(
    JaxAxisProtected[Any],
    JaxCategoricalCredalSet,
    ConvexCredalSet,
):
    """A convex hull over a finite set of categorical distributions."""

    tensor: JaxCategoricalDistribution
    protected_axes: ClassVar[dict[str, int]] = {"tensor": 1}

    def __post_init__(self) -> None:
        """Validate that the tensor contains valid categorical distributions."""
        object.__setattr__(self, "tensor", _ensure_jax_categorical_distribution(self.tensor))

    @override
    @classmethod
    def from_jax_sample(
        cls,
        sample: JaxArraySample[JaxCategoricalDistribution],
    ) -> Self:
        probabilities = _sample_probabilities(sample)
        vertices = jnp.moveaxis(probabilities, 0, -2)
        return cls(tensor=JaxProbabilityCategoricalDistribution(vertices))

    @override
    @property
    def num_classes(self) -> int:
        """Get the number of classes."""
        return self.tensor.num_classes

    def lower(self) -> jax.Array:
        """Return the per-class lower probability envelope of the convex hull."""
        return jnp.min(self.tensor.probabilities, axis=-2)

    def upper(self) -> jax.Array:
        """Return the per-class upper probability envelope of the convex hull."""
        return jnp.max(self.tensor.probabilities, axis=-2)

    @override
    @property
    def barycenter(self) -> JaxCategoricalDistribution:
        """Compute the barycenter of the convex credal set as the mean of the vertices."""
        return jax_mean(self.tensor, axis=-1)  # ty:ignore[invalid-argument-type, invalid-return-type]


@dataclass(frozen=True, slots=True, weakref_slot=True)  # ty:ignore[conflicting-metaclass]
class JaxDistanceBasedCredalSet(
    JaxAxisProtected[Any],
    JaxCategoricalCredalSet,
    DistanceBasedCredalSet,
):
    """Distance-based credal set around a nominal categorical distribution."""

    nominal: JaxCategoricalDistribution
    radius: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"nominal": 0, "radius": 0}

    def __post_init__(self) -> None:
        """Validate that nominal is a valid categorical distribution and radius is non-negative."""
        object.__setattr__(self, "nominal", _ensure_jax_categorical_distribution(self.nominal))
        object.__setattr__(self, "radius", jnp.asarray(self.radius))

    @override
    @classmethod
    def from_jax_sample(
        cls,
        sample: JaxArraySample[JaxCategoricalDistribution],
    ) -> Self:
        probabilities = _sample_probabilities(sample)
        nominal = jnp.mean(probabilities, axis=0)
        diff = jnp.abs(probabilities - nominal)
        tv_dists = 0.5 * jnp.sum(diff, axis=-1)
        radius = jnp.max(tv_dists, axis=0)
        return cls(
            nominal=JaxProbabilityCategoricalDistribution(nominal),
            radius=jnp.asarray(radius),
        )

    @override
    @property
    def num_classes(self) -> int:
        """Return the number of classes in the credal set."""
        return self.nominal.num_classes

    def lower(self) -> jax.Array:
        """Compute the lower envelope of the credal set.

        For L1/TV distance, the tightest element-wise lower bound is max(0, nominal - radius).
        """
        nominal = self.nominal.probabilities
        r = self.radius
        if r.ndim == nominal.ndim - 1:
            r = jnp.expand_dims(r, axis=-1)

        return jnp.clip(nominal - r, min=0.0, max=1.0)

    def upper(self) -> jax.Array:
        """Compute the upper envelope of the credal set.

        For L1/TV distance, the tightest element-wise upper bound is min(1, nominal + radius).
        """
        nominal = self.nominal.probabilities
        r = self.radius
        if r.ndim == nominal.ndim - 1:
            r = jnp.expand_dims(r, axis=-1)

        return jnp.clip(nominal + r, min=0.0, max=1.0)

    @override
    @property
    def barycenter(self) -> JaxCategoricalDistribution:
        """Return the nominal distribution as the barycenter of the credal set."""
        return self.nominal


_LEVEL_SET_SAMPLES = 10_000

#: Monte Carlo draws of the default (fixed-key) level set, shared between ``lower()`` and
#: ``upper()`` of the same credal set. Keyed by ``id`` of the credal set and cleaned up by a
#: finalizer, so an entry never outlives the object it belongs to.
_LEVEL_SET_SAMPLE_CACHE: dict[int, tuple[jax.Array, jax.Array]] = {}


@dataclass(frozen=True, slots=True, weakref_slot=True)  # ty:ignore[conflicting-metaclass]
class JaxDirichletLevelSetCredalSet(
    JaxAxisProtected[Any],
    JaxCategoricalCredalSet,
    DirichletLevelSetCredalSet,
):
    """Dirichlet density level set credal set.

    Contains all distributions whose Dirichlet likelihood is at least
    ``threshold`` times the peak density.
    """

    alphas: jax.Array
    threshold: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"alphas": 1, "threshold": 0}

    def __post_init__(self) -> None:
        """Validate alpha parameters and threshold."""
        object.__setattr__(self, "alphas", jnp.asarray(self.alphas))
        object.__setattr__(self, "threshold", jnp.asarray(self.threshold))

    @override
    @classmethod
    def from_jax_sample(
        cls,
        sample: JaxArraySample[JaxCategoricalDistribution],
    ) -> Self:
        """Create from a jax sample (not supported).

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

    def _log_relative_likelihood(self, samples: jax.Array) -> jax.Array:
        """Compute log relative likelihood of samples under this Dirichlet.

        Args:
            samples: Probability vectors, shape (n_samples, ..., K).

        Returns:
            Log relative likelihood, shape (n_samples, ...).
        """
        alpha = self.alphas
        alpha_m1 = alpha - 1.0

        log_density = (alpha_m1 * jnp.log(jnp.clip(samples, min=1e-30))).sum(axis=-1)

        alpha_0 = alpha.sum(axis=-1)
        k = alpha.shape[-1]
        mode = alpha_m1 / jnp.expand_dims(alpha_0 - k, axis=-1)
        mode = jnp.clip(mode, min=1e-30)
        log_mode_density = (alpha_m1 * jnp.log(mode)).sum(axis=-1)

        return log_density - log_mode_density

    def _sample_and_filter(
        self,
        n_samples: int = _LEVEL_SET_SAMPLES,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Sample from Dir(alpha) and filter to the level set.

        Args:
            n_samples: Number of samples to draw.
            key: PRNG key used for sampling. Defaults to a fixed key for determinism.

        Returns:
            Tuple of (samples, mask) where samples has shape (n_samples, ..., K)
            and mask has shape (n_samples, ...).
        """
        if key is None:
            key = jax.random.key(0)

        samples = jax.random.dirichlet(key, self.alphas, shape=(n_samples, *self.alphas.shape[:-1]))

        log_rl = self._log_relative_likelihood(samples)
        rl = jnp.exp(log_rl)

        mask = rl >= self.threshold
        return samples, mask

    def _default_sample_and_filter(self) -> tuple[jax.Array, jax.Array]:
        """Return the fixed-key Monte Carlo draws, computing them at most once per instance.

        ``lower()`` and ``upper()`` reduce over the very same draws, so the sampling pass is
        cached and shared instead of being repeated for each bound.

        Returns:
            Tuple of (samples, mask), see :meth:`_sample_and_filter`.
        """
        cache_key = id(self)
        cached = _LEVEL_SET_SAMPLE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        result = self._sample_and_filter()
        _LEVEL_SET_SAMPLE_CACHE[cache_key] = result
        weakref.finalize(self, _LEVEL_SET_SAMPLE_CACHE.pop, cache_key, None)
        return result

    def _mode_fallback(self) -> jax.Array:
        alpha_0 = self.alphas.sum(axis=-1)
        k = self.alphas.shape[-1]
        mode = jnp.clip(self.alphas - 1.0, min=0.0) / jnp.clip(jnp.expand_dims(alpha_0 - k, axis=-1), min=1.0)
        return mode / mode.sum(axis=-1, keepdims=True)

    def lower(self, key: jax.Array | None = None) -> jax.Array:
        """Compute per-class lower bounds via Monte Carlo sampling.

        Args:
            key: PRNG key used for sampling. Defaults to a fixed key for determinism.

        Returns:
            Lower probability bounds, shape (..., K).
        """
        samples, mask = self._default_sample_and_filter() if key is None else self._sample_and_filter(key=key)
        masked = jnp.where(jnp.expand_dims(mask, axis=-1), samples, jnp.inf)
        result = jnp.min(masked, axis=0)
        no_accepted = ~jnp.any(mask, axis=0)
        mode = self._mode_fallback()
        result = jnp.where(jnp.expand_dims(no_accepted, axis=-1), mode, result)
        return jnp.clip(result, min=0.0, max=1.0)

    def upper(self, key: jax.Array | None = None) -> jax.Array:
        """Compute per-class upper bounds via Monte Carlo sampling.

        Args:
            key: PRNG key used for sampling. Defaults to a fixed key for determinism.

        Returns:
            Upper probability bounds, shape (..., K).
        """
        samples, mask = self._default_sample_and_filter() if key is None else self._sample_and_filter(key=key)
        masked = jnp.where(jnp.expand_dims(mask, axis=-1), samples, -jnp.inf)
        result = jnp.max(masked, axis=0)
        no_accepted = ~jnp.any(mask, axis=0)
        mode = self._mode_fallback()
        result = jnp.where(jnp.expand_dims(no_accepted, axis=-1), mode, result)
        return jnp.clip(result, min=0.0, max=1.0)

    @override
    @property
    def barycenter(self) -> JaxCategoricalDistribution:
        """Return the Dirichlet mean as the barycenter."""
        return JaxProbabilityCategoricalDistribution(self.alphas / jnp.sum(self.alphas, axis=-1, keepdims=True))


@dataclass(frozen=True, slots=True, weakref_slot=True)  # ty:ignore[conflicting-metaclass]
class JaxProbabilityIntervalsCredalSet(
    JaxAxisProtected[Any],
    JaxCategoricalCredalSet,
    ProbabilityIntervalsCredalSet,
):
    """Credal set represented by lower/upper categorical bounds."""

    lower_bounds: jax.Array
    upper_bounds: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"lower_bounds": 1, "upper_bounds": 1}

    @override
    @classmethod
    def from_jax_sample(
        cls,
        sample: JaxArraySample[JaxCategoricalDistribution],
    ) -> Self:
        probabilities = _sample_probabilities(sample)
        lower_bounds = jnp.min(probabilities, axis=0)
        upper_bounds = jnp.max(probabilities, axis=0)
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
        del force
        stacked = jnp.stack([self.lower_bounds, self.upper_bounds], axis=-2)
        return cast("NDArray[Any]", jax.device_get(stacked))

    @override
    def __array__(self, dtype: Any = None, /, *, copy: bool | None = None) -> NDArray[Any]:
        array = self.numpy()
        if dtype is not None:
            return array.astype(dtype, copy=bool(copy))
        if copy:
            return array.copy()
        return array

    def width(self) -> jax.Array:
        """Compute interval width for each class."""
        return self.upper_bounds - self.lower_bounds

    def lower(self) -> jax.Array:
        """Return the per-class lower probability envelope (alias for ``lower_bounds``)."""
        return self.lower_bounds

    def upper(self) -> jax.Array:
        """Return the per-class upper probability envelope (alias for ``upper_bounds``)."""
        return self.upper_bounds

    def contains(self, probabilities: jax.Array) -> jax.Array:
        """Check whether probabilities are inside the intervals."""
        within_bounds = (probabilities >= self.lower_bounds) & (probabilities <= self.upper_bounds)
        return jnp.all(within_bounds, axis=-1)

    @override
    @property
    def barycenter(self) -> JaxCategoricalDistribution:
        return JaxProbabilityCategoricalDistribution(intersection_probability(self.lower_bounds, self.upper_bounds))


create_probability_intervals.register(JaxCategoricalDistribution, JaxProbabilityIntervalsCredalSet.from_sample)
create_probability_intervals.register(JaxArraySample, JaxProbabilityIntervalsCredalSet.from_jax_sample)
create_convex_credal_set.register(JaxArraySample, JaxConvexCredalSet.from_jax_sample)
create_distance_based_credal_set.register(JaxArraySample, JaxDistanceBasedCredalSet.from_jax_sample)


@create_probability_intervals_from_lower_upper_array.register(jax.Array)
def _create_probability_intervals_from_lower_upper_array(
    bounds: jax.Array,
) -> JaxProbabilityIntervalsCredalSet:
    reshaped = bounds.reshape(*bounds.shape[:-1], 2, -1)
    return JaxProbabilityIntervalsCredalSet(reshaped[..., 0, :], reshaped[..., 1, :])


@create_probability_intervals_from_bounds.register(jax.Array)
def _create_probability_intervals_from_bounds(
    probs: jax.Array, lower: jax.Array, upper: jax.Array
) -> JaxProbabilityIntervalsCredalSet:
    return JaxProbabilityIntervalsCredalSet(probs - lower, probs + upper)


def _broadcast_radius_to_batch(
    radius: jax.Array | float,
    batch_size: int,
    *,
    dtype: Any = None,  # noqa: ANN401
) -> jax.Array:
    """Promote a scalar radius to a per-sample array matching the nominal's batch dim.

    Per-sample radii are required for the protected-axes machinery to permit
    ``jax_concatenate`` across a batch of credal sets. Also aligns dtype with
    the center so downstream ``nominal - radius`` doesn't raise a dtype error.
    """
    radius_t = jnp.asarray(radius)
    if dtype is not None:
        radius_t = radius_t.astype(dtype)
    if radius_t.ndim == 0:
        radius_t = jnp.broadcast_to(radius_t, (batch_size,))
    return radius_t


@create_distance_based_credal_set_from_center_and_radius.register(jax.Array)
def _create_distance_based_credal_set_from_center_and_radius(
    center: jax.Array,
    radius: jax.Array,
) -> JaxDistanceBasedCredalSet:
    batch_size = center.shape[0] if center.ndim >= 2 else 1
    return JaxDistanceBasedCredalSet(
        nominal=JaxProbabilityCategoricalDistribution(center),
        radius=_broadcast_radius_to_batch(radius, batch_size, dtype=center.dtype),
    )


@create_distance_based_credal_set_from_center_and_radius.register(JaxCategoricalDistribution)
def _create_distance_based_credal_set_from_categorical_distribution(
    center: JaxCategoricalDistribution,
    radius: jax.Array,
) -> JaxDistanceBasedCredalSet:
    nominal_array = center.array
    batch_size = nominal_array.shape[0] if nominal_array.ndim >= 2 else 1
    return JaxDistanceBasedCredalSet(
        nominal=center,
        radius=_broadcast_radius_to_batch(radius, batch_size, dtype=nominal_array.dtype),
    )


@create_dirichlet_level_set_credal_set.register(jax.Array)
def _create_dirichlet_level_set_from_array(
    alphas: jax.Array,
    threshold: jax.Array,
) -> JaxDirichletLevelSetCredalSet:
    return JaxDirichletLevelSetCredalSet(
        alphas=alphas,
        threshold=jnp.asarray(threshold),
    )
