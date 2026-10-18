"""JAX implementations of scoring rule loss vectors."""

from __future__ import annotations

from jax import numpy as jnp

from probly.representation.sample.jax import JaxSample

from ._common import _brier_loss_vector, _log_loss_vector, _loss_at_label, _spherical_loss_vector, _zero_one_loss_vector


@_log_loss_vector.register
def jax_log_loss_vector(probabilities: jnp.ndarray) -> jnp.ndarray:
    """Compute the per-label log loss vector for a JAX array."""
    return -jnp.log(probabilities)


@_brier_loss_vector.register
def jax_brier_loss_vector(probabilities: jnp.ndarray) -> jnp.ndarray:
    """Compute the per-label Brier loss vector for a JAX array."""
    squared_norm = jnp.sum(probabilities**2, axis=-1, keepdims=True)
    return squared_norm - 2.0 * probabilities + 1.0


@_zero_one_loss_vector.register
def jax_zero_one_loss_vector(probabilities: jnp.ndarray) -> jnp.ndarray:
    """Compute the per-label zero-one loss vector for a JAX array."""
    num_classes = probabilities.shape[-1]
    argmax = jnp.argmax(probabilities, axis=-1)
    one_hot = jnp.eye(num_classes, dtype=probabilities.dtype)[argmax]
    return 1.0 - one_hot


@_spherical_loss_vector.register
def jax_spherical_loss_vector(probabilities: jnp.ndarray) -> jnp.ndarray:
    """Compute the per-label spherical loss vector for a JAX array."""
    norm = jnp.sqrt(jnp.sum(probabilities**2, axis=-1, keepdims=True))
    return 1.0 - probabilities / norm


@_loss_at_label.register
def jax_loss_at_label(loss_vector: jnp.ndarray, labels: object) -> jnp.ndarray:
    """Pick the loss of the true label for a JAX array."""
    index = jnp.asarray(labels).astype(jnp.int32)
    return jnp.take_along_axis(loss_vector, index[..., None], axis=-1)[..., 0]


@_log_loss_vector.register(JaxSample)
def _(probabilities: JaxSample) -> JaxSample:
    """Compute memberwise log loss, preserving sample metadata."""
    return JaxSample(_log_loss_vector(probabilities.array), probabilities.sample_axis, probabilities.weights)


@_brier_loss_vector.register(JaxSample)
def _(probabilities: JaxSample) -> JaxSample:
    """Compute memberwise Brier loss, preserving sample metadata."""
    return JaxSample(_brier_loss_vector(probabilities.array), probabilities.sample_axis, probabilities.weights)


@_zero_one_loss_vector.register(JaxSample)
def _(probabilities: JaxSample) -> JaxSample:
    """Compute memberwise zero-one loss, preserving sample metadata."""
    return JaxSample(_zero_one_loss_vector(probabilities.array), probabilities.sample_axis, probabilities.weights)


@_spherical_loss_vector.register(JaxSample)
def _(probabilities: JaxSample) -> JaxSample:
    """Compute memberwise spherical loss, preserving sample metadata."""
    return JaxSample(_spherical_loss_vector(probabilities.array), probabilities.sample_axis, probabilities.weights)
