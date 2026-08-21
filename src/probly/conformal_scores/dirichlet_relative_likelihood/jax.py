"""JAX implementation for Dirichlet relative likelihood scores."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from probly.representation.distribution.jax_dirichlet import JaxDirichletDistribution

from ._common import dirichlet_rl_score_func


@dirichlet_rl_score_func.register(jax.Array)
def compute_dirichlet_rl_score_jax(alphas: jax.Array, y_true: jax.Array) -> jax.Array:
    """Compute the Dirichlet relative likelihood score using JAX Arrays.

    Args:
        alphas: Dirichlet concentration parameters, shape (..., K).
        y_true: Ground truth class labels, shape (...,).
    """
    alphas_j = jnp.asarray(alphas)
    y_true_j = jnp.asarray(y_true).astype(int)
    alpha_y = jnp.take_along_axis(alphas_j, y_true_j[..., jnp.newaxis], axis=-1).squeeze(-1)
    alpha_max = jnp.max(alphas_j, axis=-1)
    return 1.0 - alpha_y / alpha_max


@dirichlet_rl_score_func.register(JaxDirichletDistribution)
def compute_dirichlet_rl_score_jax_dirichlet(dirichlet: JaxDirichletDistribution, y_true: jax.Array) -> jax.Array:
    """Compute the score from a JaxDirichletDistribution."""
    return compute_dirichlet_rl_score_jax(dirichlet.alphas, y_true)
