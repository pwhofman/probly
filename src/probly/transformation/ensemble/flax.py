"""Flax ensemble implementation."""

from __future__ import annotations

from typing import Tuple, List

from flax import nnx
import jax
import jax.numpy as jnp

from .common import register 

def _clone(obj: nnx.Module) -> Tuple[nnx.Module, dict]:
    """Deep copy of params for flax module."""
    params = nnx.clone(obj) 
    return obj, params

def _random_reset(
        obj: nnx.Module, 
        key: jax.random.PRNGKey, 
        shape: jnp.ndarray
) -> Tuple[nnx.Module, dict]:
    """Reset params for flax module."""
    params = nnx.init(key, shape)
    return obj, params

def generate_flax_ensemble(
        obj: nnx.Module, 
        num_members: int, 
        reset_params: bool
) -> List[Tuple[nnx.Module, dict]]:
    """Build a flax ensemble by initializing n_members times."""
    if reset_params:
        return [_random_reset(obj, jax.random.PRNGKey(i), jnp.ones((1,))) for i in range(num_members)]
    return [_clone(obj) for _ in range(num_members)]
   
register(nnx.Module, generate_flax_ensemble)
