"""Test dropconnect transformation with Flax models"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from flax import linen as nn

from probly.transformation.dropconnect import dropconnect


class SimpleFlaxModel(nn.Module):
    """Simple Flax model"""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=5)(x)
        x = nn.relu(x)
        x = nn.Dense(features=2)(x)
        return x



def test_dropconnect_with_flax_model() -> None:
    """Test that dropconnect transformation works correctly"""
    # Create a simple Flax model
    model = SimpleFlaxModel()

    # Initialize model parameters
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((5, 5))
    variables = model.init(key, dummy_input)

    # Apply dropconnect transformation
    transformed_model = dropconnect(model, p = 0.3)

    # Test that transformation doesnot crash
    assert transformed_model is not None

