"""Test module for Flax ensemble implementation."""

from __future__ import annotations

import logging

from flax import nnx
from jax import random

from probly.transformation.ensemble.flax import generate_flax_ensemble


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


# Create PRNG key and wrap it in nnx.Rngs
key = random.PRNGKey(0)
rngs = nnx.Rngs(params=key)

# Create model with rngs
model = SimpleModel(rngs=rngs)
ensemble = generate_flax_ensemble(model, n_members=5)
logger = logging.getLogger(__name__)
logger.info("Ensemble size: %d", len(ensemble))
print(f"Ensemble size: {len(ensemble)}")  # noqa: T201
