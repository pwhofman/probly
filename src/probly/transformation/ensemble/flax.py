"""Flax ensemble implementation."""

from __future__ import annotations

from typing import Tuple, List

from flax import nnx
import jax
import jax.numpy as jnp

from .common import register 

def _reset_clone(obj: nnx.Module, input_shape: tuple, key: jax.random.PRNGKey) -> Tuple[nnx.Module, dict]:
    """New params initialized."""
    x = jnp.ones(input_shape)
    params = obj.init(key, x)
    return obj, params

def generate_flax_ensemble(obj: nnx.Module, n_members: int, input_shape: tuple) -> List[Tuple[nnx.Module, dict]]:
    """Build a flax ensemble by initializing n_members times."""
    rng = jax.random.PRNGKey(0)
    subkeys = jax.random.split(jax.random.PRNGKey(0), n_members)
    return [_reset_clone(obj, input_shape, k) for k in subkeys]
   
register(nnx.Module, generate_flax_ensemble)