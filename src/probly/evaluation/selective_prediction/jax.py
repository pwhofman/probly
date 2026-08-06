"""JAX implementation of the selective prediction evaluation task."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._common import selective_prediction


@selective_prediction.register(jax.Array)
def selective_prediction_jax(criterion: jax.Array, losses: jax.Array, n_bins: int = 50) -> tuple[jax.Array, jax.Array]:
    """Perform selective prediction for JAX arrays."""
    if n_bins > losses.shape[0]:
        msg = "The number of bins can not be larger than the number of elements criterion"
        raise ValueError(msg)
    sort_idxs = jnp.argsort(criterion)[::-1]
    losses_sorted = losses[sort_idxs]
    bin_len = losses.shape[0] // n_bins
    bin_losses = jnp.stack([jnp.mean(losses_sorted[(i * bin_len) :]) for i in range(n_bins)])
    aurc = jnp.trapezoid(bin_losses, jnp.linspace(0, 1, n_bins))
    return aurc, bin_losses
