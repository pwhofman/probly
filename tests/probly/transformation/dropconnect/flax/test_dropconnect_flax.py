"""Simple test for dropconnect with Flax models."""

from __future__ import annotations

from flax import linen as nn
import jax.numpy as jnp

from probly.transformation.dropconnect import dropconnect


class SimpleFlaxModel(nn.Module):
    """A simple Flax model for testing."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(features=2)(x)


def test_dropconnect_flax_basic() -> None:
    """Simple test that dropconnect works with Flax models."""
    # Test that the function exists and can be called
    assert callable(dropconnect)

    # Create a simple Flax model
    model = SimpleFlaxModel()

    # Apply dropconnect
    result = dropconnect(model, p=0.3)
    assert result is not None


def test_dropconnect_flax_different_probabilities() -> None:
    """Test dropconnect with Flax models using different probabilities."""
    model = SimpleFlaxModel()

    # Test with different probability values
    for p in [0.0, 0.25, 0.5, 0.75]:
        result = dropconnect(model, p=p)
        assert result is not None
