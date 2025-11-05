from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

class DropConnectLinear(nnx.Module):
    """Custom Linear layer with DropConnect applied to weights during training.

    Attributes:
        in_features: int, Number of input features.
        out_features: int, Number of output features.
        p: float, probability of dropping individual weights.
        weight: jax.array, weight matrix of the layer.
        bias: jax.array, bias of the layer.

    """
    def __init__(self, base_layer: nnx.Linear, p: float = 0.25) -> None:
        """Initialize a DropConnectLinear layer based on a given linear base layer.

        Args:
            base_layer: nnx.Linear, The original linear layer to be wrapped.
            p: float, The probability of dropping individual weights.
        """
        self.base_layer = base_layer
        self.p = float(p)
        self.rng_key = jax.random.PRNGKey(0)

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the DropConnectLinear layer.

        Args:
            x: jnp.ndarray, input data
        Returns:
            jnp.ndarray, output layer

        """
        weight = self.base_layer.kernel
        bias = self.base_layer.bias

        training = getattr(self, 'training', False)

        if training:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            mask = jax.random.bernoulli(subkey, p=1.0 - self.p, shape = weight.shape)
            weight = weight * mask # Apply DropConnect
        else:
            weight = weight * (1 - self.p) # Scale weights at interference time

        """layer output after applying DropConnect."""
        lin_out = jnp.dot(x, weight)
        if bias is not None:
            lin_out = lin_out + bias
        return lin_out

    def extra_repr(self) -> str:
        """Expose description of in- and out-features of this layer."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.base_layer.bias is not None}"
        )
