"""Implementation For Temperature Scaling Extension Of Base Flax."""

from __future__ import annotations

from flax import nnx
import jax
from jax import numpy as jnp

from probly.calibration.scaling import common

from .flax_base import ScalerFlax


class FlaxTemperature(ScalerFlax):
    """Wrapper class for Temperature Scaling."""

    def __init__(self, base: nnx.Module, num_classes: int) -> None:
        """Initialize Wrapper with temperature.

        Args:
            base: The base model to calibrate.
            num_classes: The number of classes the model is trained on.
        """
        super().__init__(base, num_classes)
        self.temperature = nnx.Param(jnp.array(1.0, dtype=jnp.float32))

    def _scale_logits(self, logits: jax.Array) -> jax.Array:
        """Scale the logits based on learned parameters."""
        return logits / self.temperature.value

    def _init_opt_params(self) -> dict:
        """Create a dictionary with all parameters to optimize."""
        return {"t": self.temperature.value}

    def _assign_opt_params(self, params: dict) -> None:
        self.temperature.value = params["t"]

    def _loss_with_params(self, params: dict, logits: jax.Array, labels: jax.Array) -> jax.Array:
        """Loss Function for Temperature Scaling in Flax."""
        temperature = params["t"]
        z = logits / temperature

        labels = labels.astype(jnp.int32).reshape(-1)
        log_probs = z - jax.nn.logsumexp(z, axis=-1, keepdims=True)
        nll = -log_probs[jnp.arange(labels.shape[0]), labels]

        return jnp.mean(nll)


@common.register_temperature_factory(nnx.Module)
def _(_base: nnx.Module) -> type[FlaxTemperature]:
    return FlaxTemperature
