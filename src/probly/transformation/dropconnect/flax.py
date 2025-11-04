"""Flax DropConnect implementation."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp

from .common import register


def replace_flax_dropconnect(obj: nnx.Linear, p: float) -> DropConnectLinear:
    """Replace a given nnx.layer by a DropConnectLinear layer."""
    return DropConnectLinear(obj, p=p)


register(nnx.Linear, replace_flax_dropconnect)

"""probly layer mit DropConnectLineaar Implementierung für Flax."""


class DropConnectLinear(nnx.Linear):
    """A DropConnect Linear layer for Flax."""

    def __init__(self, base_layer: nnx.Linear, p: float) -> None:
        """A Dummy DropConnect Linear layer for Flax."""
        dummy_rngs = nnx.Rngs(jax.random.key(0))

        super().__init__(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rngs=dummy_rngs,
        )
        self.p = p
        self.base_layer = base_layer

    def __call__(self, x: jnp.ndarray, *, rngs: dict[str, jax.random.PRNGKey] | None = None) -> jnp.ndarray:
        """Forward pass with DropConnect applied to the weights."""
        kernel = self.base_layer.kernel
        if rngs and "dropconnect" in rngs:
            rng = rngs["dropconnect"]
            mask = jax.random.bernoulli(rng, p=1 - self.p, shape=kernel.shape)
            kernel = kernel * mask / (1 - self.p)
        return jnp.dot(x, kernel) + self.base_layer.bias
