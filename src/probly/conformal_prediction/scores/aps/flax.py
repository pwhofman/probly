""" "flax for APS."""

from __future__ import annotations

from typing import Any

import flax as jnp
import numpy as np
import numpy.typing as npt


def aps_jax(probs: Any) -> npt.NDArray[np.float64]:
    """Compute APS scores for JAX arrays."""
    sorted_probs = jnp.sort(probs, axis=1)[:, ::-1]
    cumsum_probs = jnp.cumsum(sorted_probs, axis=1)
    ranks = jnp.arange(1, probs.shape[1] + 1)
    aps_scores = jnp.sum(cumsum_probs / ranks, axis=1)
    return aps_scores


from .common import register

register("jax.numpy.ndarray", aps_jax)
