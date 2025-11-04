"""Test module for Flax ensemble implementation."""

from __future__ import annotations

import logging

from flax import nnx
from jax import random
import jax.numpy as jnp

from probly.transformation.ensemble.flax_ensemble import generate_flax_ensemble


class SimpleModel(nnx.Module):
    """Simple neural network model for testing ensemble generation."""

    def __init__(self, rngs: nnx.Rngs) -> None:
        """Initialize the model.

        Args:
            rngs: Random number generators for parameter initialization
        """
        super().__init__()
        self.t1 = nnx.Linear(2, 2, rngs=rngs)
        self.t2 = nnx.Linear(2, 2, rngs=rngs)
        self.t3 = nnx.Linear(2, 2, rngs=rngs)

    def forward(self, x: nnx.Array) -> nnx.Array:
        """Forward pass of the model.

        Args:
            x: Input tensor
        Returns:
            Output tensor after passing through layers
        """
        x = self.t1(x)
        x = self.t2(x)
        x = self.t3(x)
        return x


# --- 2. Pytest Functions ---


def test_ensemble_creation_size() -> None:
    """Tests if the correct number of ensemble members is created."""
    # ARRANGE
    key = random.PRNGKey(0)
    rngs = nnx.Rngs(params=key)
    model = SimpleModel(rngs=rngs)
    n_members = 5

    logger = logging.getLogger(__name__)
    logger.info("Testing ensemble generation with %d members...", n_members)

    # ACT
    ensemble = generate_flax_ensemble(model, n_members=n_members)

    # ASSERT
    assert len(ensemble) == n_members
    assert isinstance(ensemble[0], SimpleModel)


def test_ensemble_members_are_independent() -> None:
    """Tests if the created models have different (independent) parameters."""
    key = random.PRNGKey(42)  # Use a different key
    rngs = nnx.Rngs(params=key)
    model = SimpleModel(rngs=rngs)
    n_members = 2

    ensemble = generate_flax_ensemble(model, n_members=n_members)

    params_model_0 = ensemble[0].t1.kernel.value
    params_model_1 = ensemble[1].t1.kernel.value

    assert not jnp.array_equal(params_model_0, params_model_1)
