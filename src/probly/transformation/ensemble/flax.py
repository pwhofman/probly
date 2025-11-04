"""Flax to Ensemble implementation."""

from __future__ import annotations

from flax.nnx import Module, Param, Rngs
from jax import random

from probly.traverse_nn import nn_compose
from pytraverse import CLONE, lazydispatch_traverser, traverse

from .common import register

param_reset = lazydispatch_traverser[Param]()


@param_reset.register
def _(p: Param, rngs: Rngs) -> Param:
    key = rngs.params()
    p.value = random.normal(key, p.value.shape)
    return p


def _reset_copy(model: Module, seed: int) -> Module:
    Rngs(params=random.PRNGKey(seed))
    return traverse(
        model,
        nn_compose(param_reset),
        init={CLONE: True},
    )


def generate_flax_ensemble(obj: Module, n_members: int) -> list[Module]:
    """Stop crying ruff omfg."""
    return [_reset_copy(obj, seed=i) for i in range(n_members)]


register(Module, generate_flax_ensemble)
