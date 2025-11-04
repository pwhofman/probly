"""Ensemble flax implementation."""

from __future__ import annotations

from flax import nnx

from .common import register


def generate_flax_ensemble(obj: nnx.Module, n_members: int) -> list[nnx.Module]:  # noqa: D103
    base_model = obj.__class__
    models = []

    for x in range(n_members):
        rngs = nnx.Rngs(params=x)
        new_model = base_model(rngs=rngs)
        models.append(new_model)

    return models


register(nnx.Module, generate_flax_ensemble)
