"""JAX-based categorical distribution representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, override

import jax
import jax.numpy as jnp

from probly.representation._protected_axis.jax import JaxAxisProtected
from probly.representation.distribution._common import (
    CategoricalDistribution,
    CategoricalDistributionSample,
    create_categorical_distribution,
    create_categorical_distribution_from_logits,
)
from probly.representation.sample.jax import JaxArraySample

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import numpy as np


@create_categorical_distribution.register(jax.Array)
@dataclass(frozen=True, slots=True, weakref_slot=True)
class JaxCategoricalDistribution(
    JaxAxisProtected[Any],
    CategoricalDistribution[jax.Array],
):
    """A categorical distribution stored as a JAX array.

    Shape: (..., num_classes)
    The last axis represents the category dimension.
    """

    unnormalized_probabilities: jax.Array
    protected_axes: ClassVar[dict[str, int]] = {"unnormalized_probabilities": 1}
    # Reserved: not currently consulted because JAX has no __jax_function__-equivalent
    # dispatch hook. Kept for parity with torch/array; will activate if such a hook
    # is added in the future.
    permitted_functions: ClassVar[set[Callable]] = {jnp.mean, jnp.average}

    def __post_init__(self) -> None:
        """Validate the concentration parameters."""
        if not isinstance(self.unnormalized_probabilities, jax.Array):
            msg = "probabilities must be a JAX array."
            raise TypeError(msg)

        if self.unnormalized_probabilities.ndim < 1:
            msg = "probabilities must have at least one dimension."
            raise ValueError(msg)

        if self._is_bernoulli:
            if jnp.any(self.unnormalized_probabilities < 0) or jnp.any(self.unnormalized_probabilities > 1):
                msg = "Bernoulli probabilities must be in the range [0, 1]."
                raise ValueError(msg)
        elif jnp.any(self.unnormalized_probabilities < 0):
            msg = "Relative probabilities must be non-negative."
            raise ValueError(msg)

    @override
    def _postprocess_protected_values(self, values: dict[str, jax.Array], func: Callable) -> dict[str, jax.Array]:
        if func in (jnp.mean, jnp.average):
            # Ensure mean/average of categorical distributions uses normalized probabilities.
            values["unnormalized_probabilities"] = self.probabilities

        return values

    @override
    def _rotate_sample_axis(self, source_axis: int, dest_axis: int) -> JaxCategoricalDistribution:
        """Rotate the batch sample axis of ``unnormalized_probabilities``.

        The trailing class axis (protected) is left untouched; only batch axes
        are permuted via ``jnp.moveaxis``.
        """
        if source_axis == dest_axis:
            return self
        rotated = jnp.moveaxis(self.unnormalized_probabilities, source_axis, dest_axis)
        return JaxCategoricalDistribution(rotated)

    @property
    def _is_bernoulli(self) -> bool:
        return self.unnormalized_probabilities.shape[-1] == 1

    def _bernoulli_probability(self) -> jax.Array:
        return self.unnormalized_probabilities[..., 0]

    @override
    @property
    def probabilities(self) -> jax.Array:
        if self._is_bernoulli:
            p = self._bernoulli_probability()
            q = 1 - p
            return jnp.stack((p, q), axis=-1)

        sums = jnp.sum(self.unnormalized_probabilities, axis=-1, keepdims=True)

        return self.unnormalized_probabilities / sums

    @override
    @property
    def num_classes(self) -> int:
        if self._is_bernoulli:
            return 2
        return self.unnormalized_probabilities.shape[-1]

    @override
    @property
    def entropy(self) -> jax.Array:
        """Compute entropy along the class axis."""
        if self._is_bernoulli:
            p = self._bernoulli_probability()
            q = 1 - p
            log_p = jnp.where(p > 0, jnp.log(p), jnp.zeros_like(p))
            log_q = jnp.where(q > 0, jnp.log(q), jnp.zeros_like(q))
            return -(p * log_p + q * log_q)

        p = self.probabilities
        log_p = jnp.where(p > 0, jnp.log(p), jnp.zeros_like(p))
        return -jnp.sum(p * log_p, axis=-1)

    @override
    def sample(
        self,
        num_samples: int = 1,
        rng: jax.Array | None = None,
    ) -> JaxArraySample:
        """Sample from the categorical distribution (JAX backend).

        Args:
            num_samples: Number of samples to draw.
            rng: A ``jax.Array`` random key. JAX has no global RNG, so when ``None`` is
                passed a deterministic default key is used.

        Returns:
            A ``JaxArraySample`` of shape ``(num_samples, *batch_shape)`` with ``sample_axis=0``.
        """
        key = jax.random.key(0) if rng is None else rng

        if self._is_bernoulli:
            probabilities = self._bernoulli_probability()
            expanded = jnp.broadcast_to(probabilities, (num_samples, *probabilities.shape))
            samples = jax.random.bernoulli(key, p=expanded).astype(jnp.int32)
            return JaxArraySample(array=samples, sample_axis=0)

        flat_probabilities = self.probabilities.reshape((-1, self.num_classes))
        log_probabilities = jnp.log(flat_probabilities)
        flat_samples = jax.random.categorical(
            key, log_probabilities, axis=-1, shape=(num_samples, log_probabilities.shape[0])
        )
        samples = flat_samples.reshape((num_samples, *self.shape))
        return JaxArraySample(array=samples, sample_axis=0)

    @override
    def numpy(self, *, force: bool = False) -> np.ndarray:
        """Convert to a numpy array.

        Args:
            force: Ignored on JAX (kept for API parity with the torch backend).

        Returns:
            The probabilities as a NumPy array.
        """
        del force
        import numpy as np  # noqa: PLC0415

        return np.asarray(self.probabilities)

    @override
    def __iter__(self) -> Iterator[Any]:
        return self.probabilities.__iter__()

    @override
    def __eq__(self, value: Any) -> jax.Array:  # ty: ignore[invalid-method-override]  # noqa: PYI032
        """Vectorized equality comparison."""
        if isinstance(value, JaxCategoricalDistribution):
            eq = jnp.equal(self.probabilities, value.probabilities)
        else:
            eq = jnp.equal(self.unnormalized_probabilities, value)
        return jnp.all(eq, axis=-1)

    def __hash__(self) -> int:
        """Return an identity-based hash.

        We intentionally bypass ``super()`` here because protocol-heavy MROs can
        produce invalid ``super(type, obj)`` bindings at runtime. ``object``'s
        hash gives per-instance identity semantics.
        """
        return object.__hash__(self)


class JaxCategoricalDistributionSample(  # ty:ignore[conflicting-metaclass]
    CategoricalDistributionSample,  # intentionally parameterless: T propagates as unbound
    JaxArraySample,
):
    """Sample of categorical distributions stored as JAX arrays.

    Stores a :class:`JaxCategoricalDistribution` whose protected fields hold the
    per-sample probabilities along ``sample_axis``. Rotation of ``sample_axis``
    is handled by :meth:`JaxArraySample.samples`, which delegates to
    ``JaxCategoricalDistribution._rotate_sample_axis``.

    Note: this class intentionally does not parametrize
    ``CategoricalDistributionSample`` because :class:`JaxArraySample` is
    non-generic and binds the ``Sample`` type parameter to ``jax.Array``;
    the unbound type variable from the parent class is left unbound rather
    than dropped from the MRO. Keeping the generic argument here would
    produce inconsistent ``Sample`` type arguments across the MRO.
    """

    sample_space: ClassVar[type[CategoricalDistribution]] = JaxCategoricalDistribution

    @override
    @classmethod
    def __instancehook__(cls, instance: object) -> bool:
        return super().__instancehook__(instance)


@create_categorical_distribution.register(JaxCategoricalDistribution)
def _create_jax_categorical_distribution_from_instance(
    data: JaxCategoricalDistribution,
) -> JaxCategoricalDistribution:
    return data


@create_categorical_distribution_from_logits.register(jax.Array)
def _create_jax_categorical_distribution_from_logits(
    data: jax.Array,
) -> JaxCategoricalDistribution:
    return JaxCategoricalDistribution(jax.nn.softmax(data, axis=-1))
