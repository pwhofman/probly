"Ensemble flax implementation"

from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx

from .common import register

if TYPE_CHECKING:
    from collections.abc import Callable

def generate_flax_ensemble(obj: nnx.Module, n_members: int) -> list[nnx.Module]:
    base_model = obj.__class__
    rngs = nnx.Rngs(params=0)
    my_model = base_model(rngs=rngs)
    models = []

    for x in range(n_members):
        rngs = nnx.Rngs(params=x)
        new_model = base_model(rngs=rngs)
        models.append(new_model)

    return models

register(nnx.Module, generate_flax_ensemble)