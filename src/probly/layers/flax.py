"""Flax (nnx) DropConnect Linear Layer Implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax 

from flax import nnx

from jax.numpy import jnp


class DropConnectLinear(nnx.Module):
    """A linear layer with DropConnect regularization for Flax (nnx)."""

    def __init__ (self, base_layer: nnx.Linear, p: float= 0.25)-> None:
        """Initialize the DropConnectLinear layer.
        
        Args:
            base_layer: The Original nnx.Linear layer to apply DropConnect to.
            p: The probability of dropping out individual weights."""
        
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.p = p
        self.weight = base_layer.weight     ##weight und bias muss nicht geklont werden
        self.bias = base_layer.bias

    def __call__(
        self, 
        x: jax.Array,
        *,
        training: bool = True,
        key: jax.Array | None = None
    ) -> jax.Array:
        """Forward pass applying DropConnect to the weights during training.""" ##docstring hat auch eine Syntax ist nicht wie ein Kommentar

        if training:
            if key is None:
                raise ValueError("A prng key must be provided during training for DropConnect.")
            mask = jax.random.bernoulli(key, 1 - self.p, self.weight.shape)
            weight = self.weight * mask
        else:
            weight = self.weight * (1 - self.p)
        y = jnp.dot(x, weight.T)
        if self.bias is not None:
            y += self.bias
        return y  
