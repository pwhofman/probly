"""Flax ensemble implementation."""

from __future__ import annotations

from flax.nnx import Conv, Linear, Module, Rngs
import jax.random

from probly.traverse_nn import nn_compose
from pytraverse import CLONE, lazydispatch_traverser, traverse

from .common import register

reset_traverser_nnx = lazydispatch_traverser[object](name="reset_traverser_nnx")


@reset_traverser_nnx.register
def _(obj: Linear, rngs: Rngs) -> Module:
    """Register for a Linear Layer."""
    # Pulling the rngs.
    rng_key = rngs.get("params", jax.random.PRNGKey(0))  # Fallback if "params" does not exist

    # Splitting into new keys.
    rng_key, subkey = jax.random.split(rng_key)

    # initialize with new random input.
    obj.__init__(obj.in_features, obj.out_features, rngs=subkey)

    return obj


@reset_traverser_nnx.register
def _(obj: Conv, rngs: Rngs) -> Module:
    """Register for a Conv Layer."""
    # Pulling the rngs.
    rng_key = rngs.get("params", jax.random.PRNGKey(0))  # Fallback if "params" does not exist

    # Splitting into new keys.
    rng_key, subkey = jax.random.split(rng_key)

    # initialize with new random input.
    obj.__init__(obj.in_features, obj.out_features, obj.kernel_size, obj.padding, rngs=subkey)
    return obj


def _reset_copy_nnx(module: Module) -> Module:
    return traverse(module, nn_compose(reset_traverser_nnx), init={CLONE: True})


def generate_flax_nnx_ensemble(obj: Module, n_members: int) -> list[Module]:
    """Build a flax ensemble by copying the base model n_members times."""
    return [_reset_copy_nnx(obj) for _ in range(n_members)]


register(Module, generate_flax_nnx_ensemble)
