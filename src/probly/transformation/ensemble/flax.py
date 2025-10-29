"""Flax ensemble implementation."""
from __future__ import annotations

from flax.nnx import Module, Rngs
import jax

from probly.traverse_nn import nn_compose
from pytraverse import CLONE, lazydispatch_traverser, traverse

from .common import register

reset_traverser_nnx = lazydispatch_traverser[object](name="reset_traverser_nnx")

@reset_traverser_nnx.register
def _(obj: Module, rngs: Rngs | None = None) -> Module:
    rngs = rngs or Rngs(jax.random.key(0))
    return type(obj)(*getattr(obj, "args", ()), rngs=rngs, **getattr(obj, "kwargs", {}))

def _reset_copy_nnx(module: Module) -> Module:
    return traverse(module, nn_compose(reset_traverser_nnx), init={CLONE: True})

def generate_flax_nnx_ensemble(obj: Module, n_members: int) -> list[Module]:
    """Build a flax ensemble by copying the base model n_members times."""
    return [_reset_copy_nnx(obj) for _ in range(n_members)]

register(Module, generate_flax_nnx_ensemble)
