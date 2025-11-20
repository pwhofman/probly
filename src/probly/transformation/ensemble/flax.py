"""Flax nnx ensemble implementation."""

from __future__ import annotations

import copy

from flax import nnx
import jax
import numpy as np

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
            rng, subkey = jax.random.split(rng)
            shape = layer.bias.value.shape
            layer.bias.value = jax.random.normal(subkey, shape)
    return module


def ensemble(
    module: nnx.Module,
    num_members: int,
    reset_params: bool = True,
    seed: int | None = None,
) -> list[nnx.Module]:
    """Create an ensemble of num_members independent clones of the given nnx.Module.

       Each clone has freshly initialized weights.

    Args:
        module: nnx.Module instance to clone
        num_members: number of ensemble members
        reset_params: Whether to reinitialize parameters (default: True).
        seed: Random seed. If None, a random seed from numpy is generated.

    Returns:
        List of nnx.Module instances with independent parameters
    """
    if not isinstance(num_members, int):
        msg = "num_members must be an int"
        raise TypeError(msg)

    if num_members < 0:
        msg = "num_members must be non-negative"
        raise ValueError(msg)

    if seed is None:
        max_seed = 2**31 - 1

        # create local generator to avoid side-effects
        rng = np.random.default_rng()
        seed = rng.integers(0, max_seed)

    base_rng = jax.random.PRNGKey(seed)
    rngs = jax.random.split(base_rng, num_members)

    clones = []
    for rng in rngs:
        new_mod = copy.deepcopy(module)
        if reset_params:
            new_mod = _reinit_nnx_module(new_mod, rng)
        clones.append(new_mod)

    return clones


register(nnx.Module, ensemble)
