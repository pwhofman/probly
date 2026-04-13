"""Conformal prediction methods for classification implemented in JAX/Flax."""

from __future__ import annotations

from flax import nnx
import jax.numpy as jnp

from ._common import conformal_generator, to_probabilities


@conformal_generator.register(nnx.Module)
def _(model: nnx.Module) -> nnx.Module:
    """Conformalise a Flax model."""
    model.conformal_quantile = None  # ty: ignore[unresolved-attribute]
    model.non_conformity_score = None  # ty: ignore[unresolved-attribute]
    return model


@to_probabilities.register(jnp.ndarray)
def _(pred: jnp.ndarray) -> jnp.ndarray:
    """Obtain probabilities from a Flax model."""
    if pred.ndim != 2:
        msg = f"Probability extraction expects a 2D array, got {pred.ndim}D array instead."
        raise ValueError(msg)
    if jnp.allclose(pred.sum(axis=-1), jnp.ones(pred.shape[0], device=pred.device)):
        # If the predictions already sum to 1, we assume they are probabilities
        return pred
    probs = nnx.softmax(pred, axis=-1)
    return probs
