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
    #The original model
    model = MLP(2, 16, 5, rngs=nnx.Rngs(0))
    original_linear1_weight_values = model.linear1.w.value

    #Generate 3 models with re-initialized parameters
    ensemble_models = generate_flax_ensemble(model, 3)

    #check the number of models
    assert len(ensemble_models) == 3

    #Check output shapes
    x = jnp.ones((1, 2))

    linear1_weight_values = []
    for p in ensemble_models:
        y = p(x)
        #because x has shape (1, 2) : 1 sample 2 features
        #And MLP has Shape (2, 16, 5) : input has 2 features, hidden layer has 16 Neurons and output layer has 5 neurons
        #That's why we should check the output whether it's (1, 5) shaped
        assert y.shape == (1, 5) 

        #make sure the new model is not the original object
        assert p is not model
        
        #Save the current linear1 weights of this model for later comparison
        weight_value = p.linear1.w.value
        linear1_weight_values.append(weight_value)

    #Ensure that each model in the ensemble has weights different from the original model
    #Verify that all ensemble models have independently initialized linear1 weights
    assert not np.array_equal(linear1_weight_values[0], original_linear1_weight_values)
    assert not np.array_equal(linear1_weight_values[1], original_linear1_weight_values)
    assert not np.array_equal(linear1_weight_values[2], original_linear1_weight_values)
