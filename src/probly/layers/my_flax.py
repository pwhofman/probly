"""probly layer mit DropConnectLinear Implementierung für Flax."""

from __future__ import annotations

from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp


class DropConnectLinear(nnx.Linear):
    """A DropConnect Linear layer for Flax."""

    def __init__(self, base_layer: nnx.Linear, p: float, **kwargs: Any) -> None:  # noqa: ANN401
        """Initializes a DropConnect Linear layer for Flax.

        Args:
            base_layer (nnx.Linear): The original linear layer from which
                weights and dimensions are taken.
            p (float): The dropout probability of the DropConnect layer.
            **kwargs: Additional keyword arguments passed to nnx.Linear.__init__.
        """
        key_dict = {"params": jax.random.PRNGKey(0)}
        dummy_rngs = nnx.Rngs(key_dict)

        super().__init__(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rngs=dummy_rngs,
            **kwargs,
        )

        self.kernel = base_layer.kernel
        self.bias = base_layer.bias

        self.p = p
        self.base_layer = base_layer

    def __call__(self, x: jnp.ndarray, *, rngs: dict[str, jax.random.PRNGKey] | None = None) -> jnp.ndarray:
        """Forward pass with DropConnect applied to the weights.

        Args:
            x (jnp.ndarray): Input array.
            rngs (dict[str, jax.random.PRNGKey] | None): Dictionary of RNG keys.
                The 'dropconnect' key is required during training.

        Returns:
            jnp.ndarray: Layer output.
        """
        kernel = self.kernel
        bias = self.bias

        if rngs and "dropconnect" in rngs:
            rng = rngs["dropconnect"]
            mask = jax.random.bernoulli(rng, p=1 - self.p, shape=kernel.shape)
            kernel = kernel * mask / (1 - self.p)

        return jnp.dot(x, kernel) + bias
