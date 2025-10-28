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
    return reset_parameter(new_model, rng_key)


def generate_flax_ensemble(model: nnx.Module, n_members: int): 
    """Build a flax ensemble by copying the base model n_members times.
    
        Args: 
            model (nnx.Module): The base model to duplicate. 
            n_members (int)   : The number of enssemble members to create. 

        Returns: 
            list of nnx.Module: A list of models, each a randomized copy of the base model.
    """
    return [_reset_and_copy(model) for _ in range(n_members)]


# In[142]:

if __name__ == "__main__":
    class MLP(nnx.Module): 
        def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs): 
            self.linear1 = Linear(din, dmid, rngs=rngs)
            self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
            self.bn = nnx.BatchNorm(dmid, rngs=rngs)
            self.linear2 = Linear(dmid, dout, rngs=rngs)

        def __call__(self, x: jax.array): 
            x = nnx.gelu(self.dropout(self.bn(self.linear1(x))))
            return self.linear2(x)


    # In[139]:


    model = MLP(2, 16, 5, rngs=nnx.Rngs(0))
    y = model(x=jnp.ones((3, 2)))
    nnx.display(model)

    #Creating 3 copies with re-initialized parameters
    ensemble = generate_flax_ensemble(model, 3)


