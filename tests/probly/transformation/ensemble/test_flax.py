#Bitte die hier nochmal checken
from probly.transformation.ensemble.flax import generate_flax_ensemble
import pytest

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

import warnings 
warnings.filterwarnings("ignore")

#Define a simple Linear layer
class Linear(nnx.Module):
  def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
    self.w = nnx.Param(rngs.params.uniform((din, dout))) # Initialize weight
    self.b = nnx.Param(jnp.zeros((dout,))) # Initialize bias vector with zeros
    self.din, self.dout = din, dout

  def __call__(self, x: jax.Array):
    return x @ self.w + self.b
    
#Define a simple multi-layer perceptron
class MLP(nnx.Module): 
    def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs): 
        self.linear1 = Linear(din, dmid, rngs=rngs) 
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs) 
        self.bn = nnx.BatchNorm(dmid, rngs=rngs)
        self.linear2 = Linear(dmid, dout, rngs=rngs)

    def __call__(self, x: jax.Array): 
        x = nnx.gelu(self.dropout(self.bn(self.linear1(x))))
        return self.linear2(x)

def test_generate_flax_ensemble():
    model = MLP(2, 16, 5, rngs=nnx.Rngs(0))
    y = model(x=jnp.ones((3, 2)))
    nnx.display(model)

    generate_flax_ensemble(model, 3)