"""Implementation For Platt Scaling in Flax."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp

from probly.calibration.scaling import common

from .flax_base import ScalerFlax


class FlaxPlatt(ScalerFlax):
    """Platt Scaling Implementation with Flax."""

    def __init__(self, base: nnx.Module) -> None:
        """Initialize Wrapper with w and biases.

        Args:
            base: The base model which outputs should be calibrated.
        """
        super().__init__(base, num_classes=1)

        self.w = nnx.Param(jnp.ones((), dtype=jnp.float32))
        self.b = nnx.Param(jnp.zeros((), dtype=jnp.float32))

    def _scale_logits(self, logits: jax.Array) -> jax.Array:
        """Scaling the logits based on learned parameters."""
        if logits.ndim == 1:
            logits = logits[..., None]

        if logits.ndim != 2 or logits.shape[-1] != 1:
            msg = "platt scaling expects logits shape (N,) or (N,1)."
            raise ValueError(msg)

        return logits * self.w.value + self.b.value

    def _init_opt_params(self) -> dict:
        """Generate dictionary with parameters for optimization."""
        return {"w": self.w.value, "b": self.b.value}

    def _assign_opt_params(self, params: dict) -> None:
        self.w.value = params["w"]
        self.b.value = params["b"]

    def _loss_with_params(self, params: dict, logits: jax.Array, labels: jax.Array) -> jax.Array:
        """Calculates loss with parameters as argument."""
        w = params["w"]
        b = params["b"]

        if logits.ndim == 1:
            logits = logits[..., None]

        if logits.ndim != 2 or logits.shape[-1] != 1:
            msg = "platt scaling expects logits shape (N,) or (N,1)."
            raise ValueError(msg)

        z = logits * w + b
        labels = labels.astype(jnp.float32).reshape(z.shape)

        loss = jnp.mean(jax.nn.softplus(z) - z * labels)

        return loss


@common.register_platt_factory(nnx.Module)
def _(_base: nnx.Module) -> type[FlaxPlatt]:
    return FlaxPlatt
