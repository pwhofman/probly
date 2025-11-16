"""Flax layer implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx
import jax

if TYPE_CHECKING:
    from flax.nnx import rnglib


class DropConnectDense(nnx.Module):
    """Custom Linear layer with DropConnect applied to weights during training.

    This implementation follows the Flax nnx.Dropconnect design pattern.
    """

    def __init__(
        self,
        base_layer: nnx.Linear,
        rate: float = 0.25,
        *,
        rng_collection: str = "dropout",
        rngs: rnglib.Rngs | rnglib.RngStream | jax.Array | None = None,
    ) -> None:
        """Initialize DropConnectDense layer.

        Args:
            base_layer: The base Linear layer to apply DropConnect to.
            rate: The probability of dropping an individual weight connection.
            rng_collection: The RNG collection name for sampling randomness.
            rngs: Optional RNG state. Can be Rngs, RngStream, or None.
        """
        self.rngs = rngs
        self.rng_collection = rng_collection
        self.in_features = base_layer.kernel.shape[0]
        self.out_features = base_layer.kernel.shape[1]
        self.rate = rate

        self.weight = base_layer.kernel.T
        self.bias = base_layer.bias if base_layer.use_bias else None

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = False,
        rngs: rnglib.Rngs | rnglib.RngStream | jax.Array | None = None,
    ) -> jax.Array:
        """Apply DropConnect to the forward pass.

        Args:
            x: Input array.
            deterministic: If True, disable dropout (use expected value).
            rngs: Optional RNG state for this call. Overrides constructor rngs.

        Returns:
            Output array after applying DropConnect.
        """
        if not deterministic:
            # Use call-time rngs if provided, otherwise fall back to constructor rngs
            rng_source = rngs if rngs is not None else self.rngs

            if rng_source is None:
                msg = "RNGs must be provided either at construction or call time"
                raise ValueError(msg)

            # Get the appropriate key based on rng_source type
            key = getattr(rng_source, self.rng_collection)() if isinstance(rng_source, nnx.Rngs) else rng_source

            mask = jax.random.uniform(key, shape=self.weight.value.shape) > self.rate
            weight = self.weight.value * mask
        else:
            weight = self.weight.value * (1.0 - self.rate)

        y = x @ weight.T
        if self.bias is not None:
            y = y + self.bias.value

        return y

    @property
    def p(self) -> float:
        """Backward compatibility."""
        return self.rate
