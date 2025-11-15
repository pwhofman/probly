#!/usr/bin/env python
# coding: utf-8

# In[119]:


from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import warnings 
warnings.filterwarnings("ignore")


# In[141]:


def reset_parameter(model: nnx.Module, rng_key): 
    """
    Reset all nnx.Param parameters in a given model using new random values.

    Args: 
        model (nnx.Module): The model whose parameters will be reset.
        rng_key           : JAX random key used to generate new parameter values.

    Returns: 
        Updated rng_key after all parameters have been reset.
    """
    for attr_name in dir(model): 
        attr = getattr(model, attr_name)
        if isinstance(attr, nnx.Param): 
            rng_key, subkey = jax.random.split(rng_key)
            rngs = nnx.Rngs(params=subkey)
            model.__dict__[attr_name] = nnx.Param(rngs.params.uniform(attr.value.shape))
        elif isinstance(attr, nnx.Module): 
            rng_key = reset_parameter(attr, rng_key)
    return rng_key


def _reset_and_copy(model: nnx.Module): 
    """
    Create a deep copy of a amodel with freshly reinitialiazed parameters. 

    Args: 
        model (nnx.Module): The base model to copy and reset.

    Returns: 
        A new nnx.Module instance with the same structure but re-initialized parameters. 
    """
    graph, state = nnx.split(model)
    new_model = nnx.merge(graph, state)
    rng_key = jax.random.PRNGKey(np.random.randint(0, 10000))
    reset_parameter(new_model, rng_key)
    return new_model


def generate_flax_ensemble(model: nnx.Module, n_members: int): 
    """Build a flax ensemble by copying the base model n_members times.
    
        Args: 
            model (nnx.Module): The base model to duplicate. 
            n_members (int)   : The number of enssemble members to create. 

        Returns: 
            list of nnx.Module: A list of models, each a randomized copy of the base model.
    """
    return [_reset_and_copy(model) for _ in range(n_members)]

