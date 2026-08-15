"""JAX implementations of scoring rule loss vectors."""

from __future__ import annotations

from jax import numpy as jnp

from ._common import _brier_loss_vector, _log_loss_vector, _spherical_loss_vector, _zero_one_loss_vector


@_log_loss_vector.register
def array_log_loss_vector(probabilities: jnp.ndarray) -> jnp.ndarray:
    """Compute the per-label log loss vector for a JAX array."""
    return -jnp.log(probabilities)


@_brier_loss_vector.register
def array_brier_loss_vector(probabilities: jnp.ndarray) -> jnp.ndarray:
    """Compute the per-label Brier loss vector for a JAX array."""
    squared_norm = jnp.sum(probabilities**2, axis=-1, keepdims=True)
    return squared_norm - 2.0 * probabilities + 1.0


@_zero_one_loss_vector.register
def array_zero_one_loss_vector(probabilities: jnp.ndarray) -> jnp.ndarray:
    """Compute the per-label zero-one loss vector for a JAX array."""
    num_classes = probabilities.shape[-1]
    argmax = jnp.argmax(probabilities, axis=-1)
    one_hot = jnp.eye(num_classes, dtype=probabilities.dtype)[argmax]
    return 1.0 - one_hot


@_spherical_loss_vector.register
def array_spherical_loss_vector(probabilities: jnp.ndarray) -> jnp.ndarray:
    """Compute the per-label spherical loss vector for a JAX array."""
    norm = jnp.sqrt(jnp.sum(probabilities**2, axis=-1, keepdims=True))
    return 1.0 - probabilities / norm
