from __future__ import annotations

import jax
from flax import nnx
from flax.nnx import rnglib   


class DropConnectDense(nnx.Module):
 
 def __init__(self, base_layer: nnx.Dense, rngs: rnglib.Rngs, p: float = 0.25):
    """Custom Linear layer with DropConnect applied to weights during training.

    Attributes:
        rngs:           flax.nnx.Rngs, The JAX RNG state abstraction.
        in_features:    int, The number of input features for the layer.
        out_features:   int, The number of output features for the layer.
        p:              float, The probability of dropping an individual weight connection.
        weight:         flax.nnx.Param, The weight matrix of the layer with shape `(out_features, in_features)`.
        bias:           flax.nnx.Param | None, The bias vector of the layer, or None if it is not used.

    """
    self.rngs = rngs
    self.in_features = base_layer.kernel.shape[0]
    self.out_features = base_layer.kernel.shape[1]
    self.p = p
    self.weight = nnx.Param(base_layer.kernel.T)

    if base_layer.use_bias:
        self.bias = nnx.Param(base_layer.bias)
    else:
        self.bias = None
    
 def __call__(self, x: jax.Array, *, deterministic: bool = False) -> jax.Array:
     if not deterministic:
        key = self.rngs.dropout()
        mask = jax.random.uniform(key, shape=self.weight.value.shape) > self.p
        weight = self.weight.value * mask  
     else:
        weight = self.weight.value * (1.0 - self.p)
     y = x @ weight.T
     if self.bias is not None:
            y = y + self.bias.value
        
     return y
     