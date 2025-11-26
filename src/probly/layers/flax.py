"""flax layer implementation."""

from __future__ import annotations

from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import first_from
import jax
from jax import lax, random
import jax.numpy as jnp


class DropConnectLinear(nnx.Module):
    """Custom Linear layer with DropConnect applied to weights during training.

    Attributes:
        weight: nnx.Param, weight matrix of shape.
        bias: nnx.Param, bias vector of shape.
        rate: float, the dropconnect probability.
        deterministic: bool, if false the inputs are masked, whereas if true, no mask
            is applied and the inputs are returned as is.
        rng_collection: str, the rng collection name to use when requesting a rng key.
        rngs: nnx.Rngs or nnx.RngStream or None, rng key.

    """

    def __init__(
        self,
        base_layer: nnx.Linear,
        rate: float = 0.25,
        *,
        rng_collection: str = "dropconnect",
        rngs: rnglib.Rngs | rnglib.RngStream | None = None,
    ) -> None:
        """Initialize a DropConnectLinear layer based on a given linear base layer.

        Args:
            base_layer: nnx.Linear, The original linear layer to be wrapped.
            rate: float, the dropconnect probability.
            rng_collection: str, rng collection name to use when requesting a rng key.
            rngs: nnx.Rngs or nn.RngStream or None, rng key.
        """
        self.weight = base_layer.kernel
        self.bias = base_layer.bias if base_layer.bias is not None else None
        self.rate = rate
        self.rng_collection = rng_collection

        if isinstance(rngs, rnglib.Rngs):
            self.rngs = rngs[self.rng_collection].fork()
        elif isinstance(rngs, rnglib.RngStream):
            self.rngs = rngs.fork()
        elif rngs is None:
            self.rngs = nnx.data(None)
        else:
            msg = f"rngs must be a RNGS, RngStream or None, but got {type(rngs)}."
            raise TypeError(msg)

    def __call__(
        self,
        inputs: jax.Array,
        *,
        deterministic: bool = False,
        rngs: rnglib.Rngs | rnglib.RngStream | jax.Array | None = None,
    ) -> jax.Array:
        """Forward pass of the DropConnectLinear layer.

        Args:
           inputs: jax.Array, input data that should be randomly masked.
           deterministic: bool, if false the inputs are masked, whereas if true, no mask
            is applied and the inputs are returned as is.
           rngs: nnx.Rngs, nnx.RngStream or jax.Array, optional key used to generate the dropconnect mask.

        Returns:
            jax.Array, layer output.
        """
        self.deterministic = deterministic

        deterministic = first_from(
            deterministic,
            self.deterministic,
            error_msg="""No `deterministic` argument was provided to DropConnect
                as either a __call__ argument or class attribute.""",
        )

        if (self.rate == 0.0) or deterministic:
            return inputs

        # Prevent gradient NaNs in 1.0 edge-case.
        if self.rate == 1.0:
            out = inputs @ jnp.zeros_like(self.weight.value)
            return out if self.bias is None else out + self.bias

        rngs = first_from(
            rngs,
            self.rngs,
            error_msg="""`deterministic` is False, but no `rngs` argument was provided
                to DropConnect as either a __call__ argument or class atribute.""",
        )

        if isinstance(rngs, rnglib.Rngs):
            key = rngs[self.rng_collection]()
        elif isinstance(rngs, rnglib.RngStream):
            key = rngs()
        elif isinstance(rngs, jax.Array):
            key = rngs
        else:
            msg = f"rngs must be Rngs, RngStream or jax.Array, but got {type(rngs)}."
            raise TypeError(msg)

        keep_prob = 1.0 - self.rate
        mask = random.bernoulli(key, p=keep_prob, shape=self.weight.value.shape)
        masked_weight = lax.select(mask, self.weight.value / keep_prob, jnp.zeros_like(self.weight.value))

        out = inputs @ masked_weight
        if self.bias is not None:
            out = out + self.bias.value
        return out

    def __repr__(self) -> str:
        """Return a string representation of the layer including its class name and key attributes."""
        return f"{self.__class__.__name__}({self.extra_repr()})"

    def extra_repr(self) -> str:
        """Expose description of in- and out-features of this layer."""
        in_features = self.weight.value.shape[0]
        out_features = self.weight.value.shape[1]
        return f"in_features={in_features}, out_features={out_features}, bias={self.bias is not None}"
