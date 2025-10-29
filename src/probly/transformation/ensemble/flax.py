"""Flax ensemble implementation."""

from __future__ import annotations

from typing import Tuple, List

from flax import nnx
import jax
import jax.numpy as jnp

from .common import register 

def _reset_clone(obj: nnx.Module, input_shape: tuple, key: jax.random.PRNGKey) -> Tuple[nnx.Module, dict]:
    """New params initialized.
    
    Args:
        obj: nnx.Module, The model to be cloned.
        input_shape: tuple, The shape of the input data.
        key: jax.random.PRNGKey, The random key for initialization.
        
    Returns:
        Tuple[nnx.Module, dict], The cloned model and its parameters.
    """
    x = jnp.ones(input_shape)
    params = obj.init(key, x)
    return obj, params

def generate_flax_ensemble(obj: nnx.Module, n_members: int, input_shape: tuple, key: jax.random.PRNGKey) -> List[Tuple[nnx.Module, dict]]:
    """Build a flax ensemble by initializing n_members times.
    
    Args:
        obj: nnx.Module, The base model to be used for the ensemble.
        n_members: int, The number of members in the ensemble.
        input_shape: tuple, The shape of the input data.
        key: jax.random.PRNGKey, The random key for initialization.
    
    Returns:
        List[Tuple[nnx.Module, dict]], The list of ensemble members with their parameters.
    """
    subkeys = jax.random.split(key, n_members)
    return [_reset_clone(obj, input_shape, k) for k in subkeys]
   
register(nnx.Module, generate_flax_ensemble)