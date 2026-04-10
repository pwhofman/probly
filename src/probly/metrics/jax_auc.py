"""JAX implementation of AUC."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from probly.metrics import auc


@auc.register(jax.Array)
def auc_jax(x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute area under a curve using the trapezoid rule."""
    return jnp.trapezoid(y, x, axis=-1)
