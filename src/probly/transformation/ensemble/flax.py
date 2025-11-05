"""Flax nnx ensemble implementation."""

from __future__ import annotations

import contextlib
import copy

from flax import nnx
import jax
import jax.numpy as jnp

from .common import register


def _reinit_nnx_module(module: nnx.Module, rng: jax.random.KeyArray) -> nnx.Module:
    """Reinitialize all parameters of a nnx.Module (stateful flax.nnx.Module).

    Currently handles Dense-like layers with `.kernel.value` and `.bias.value`.
    """
    for layer in getattr(module, "layers", []):
        # Kernel reinit
        if hasattr(layer, "kernel"):
            rng, subkey = jax.random.split(rng)
            shape = layer.kernel.value.shape
            layer.kernel.value = jax.random.normal(subkey, shape)
        # Bias reset to zeros
        if hasattr(layer, "bias"):
            layer.bias.value = jnp.zeros_like(layer.bias.value)
    return module


def ensemble(module: nnx.Module, n_members: int) -> list[nnx.Module]:
    """Create an ensemble of n_members independent clones of the given nnx.Module.

    Each clone has freshly initialized weights.

    Args:
        module: nnx.Module instance to clone
        n_members: number of ensemble members

    Returns:
        List of nnx.Module instances with independent parameters
    """
    if not isinstance(n_members, int):
        msg = "n_members must be an int"
        raise TypeError(msg)

    base_rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(base_rng, n_members)

    clones = []
    for rng in rngs:
        new_mod = copy.deepcopy(module)
        new_mod = _reinit_nnx_module(new_mod, rng)
        clones.append(new_mod)

    return clones


with contextlib.suppress(Exception):
    register(nnx.Module, ensemble)
