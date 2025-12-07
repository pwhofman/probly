"""Ensemble flax implementation."""

from __future__ import annotations

from flax import nnx
import jax

from .common import register


def generate_flax_ensemble(  # noqa: D103
    obj: nnx.Module,
    n_members: int,
    reset_params: bool = False,
) -> list[nnx.Module]:
    models: list[nnx.Module] = []
    base_rng = jax.random.key(0)
    keys = jax.random.split(base_rng, n_members)

    for k in keys:
        model = nnx.clone(obj)

        if reset_params:
            rng = k

            # iterate through the layers of the Sequential fixture
            for layer in model.layers:
                state = nnx.state(layer)

                # Reset kernel
                rng, sub = jax.random.split(rng)
                state["kernel"] = nnx.Param(jax.random.normal(sub, state["kernel"].value.shape))

                # Reset bias
                rng, sub = jax.random.split(rng)
                state["bias"] = nnx.Param(jax.random.normal(sub, state["bias"].value.shape))

                nnx.update(layer, state)

        models.append(model)

    return models


register(nnx.Module, generate_flax_ensemble)
