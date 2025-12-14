"""Simple test for dropconnect with Flax models."""

from __future__ import annotations

import pytest

pytest.importorskip("flax")

from flax import linen as nn
import jax.numpy as jnp

from probly.transformation.dropconnect import dropconnect


class SimpleFlaxModel(nn.Module):
    """A simple Flax model for testing."""

    features: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(features=self.features)(x)


@pytest.mark.parametrize("p_value", [0.0, 0.25, 0.5, 0.75])
def test_dropconnect_flax_different_probabilities(p_value: float) -> None:
    """Test dropconnect with Flax models using different probabilities."""
    model = SimpleFlaxModel()
    result = dropconnect(model, p=p_value)
    assert result is not None


def test_dropconnect_flax_basic() -> None:
    """Test that dropconnect works with Flax models."""
    # Test that the function exists and can be called
    assert callable(dropconnect)

    model = SimpleFlaxModel()

    # Apply dropconnect - should not crash
    result = dropconnect(model, p=0.3)
    assert result is not None
