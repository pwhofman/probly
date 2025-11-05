"""Ensemble flax implementation."""

from __future__ import annotations

from flax import nnx
import jax

from .common import register


def generate_flax_ensemble(obj: nnx.Module, n_members: int) -> list[nnx.Module]:  # noqa: D103
    models = []
    # Creates a base random key
    rng = jax.random.key(0)
    # Split the base key into n independent RNG subkeys ensuring each model has a different key
    rngs_list = jax.random.split(rng, n_members)
    for x in rngs_list:
        # Initialize a new model with its own random parameters
        rngs = nnx.Rngs(params=x)
        new_model = type(obj)(rngs=rngs)
        models.append(new_model)
    return models

register(nnx.Module, generate_flax_ensemble)
