"""Flax ensemble implementation."""
from __future__ import annotations

from flax.nnx import Module, Rngs
from jax import random

from probly.traverse_nn import nn_compose
from pytraverse import CLONE, lazydispatch_traverser, traverse


from .common import register

reset_traverser_nnx = lazydispatch_traverser[object](name="reset_traverser_nnx")

@reset_traverser_nnx.register
def _(obj: Module, rngs: Rngs) -> Module:
    # Linear Layer
    if hasattr(obj, "kernel") and hasattr(obj, "bias"):
        key_kernel, key_bias = random.split(rngs.params())
        obj.kernel.value = random.uniform(key_kernel, obj.kernel.shape)
        obj.bias.value = random.uniform(key_bias, obj.bias.shape)
    return obj

def _reset_copy_nnx(module: Module) -> Module:
    return traverse(module, nn_compose(reset_traverser_nnx), init={CLONE: True})

def generate_flax_nnx_ensemble(obj: Module, n_members: int) -> list[Module]:
    """Build a flax ensemble by copying the base model n_members times."""
    return [_reset_copy_nnx(obj) for _ in range(n_members)]

register(Module, generate_flax_nnx_ensemble)
