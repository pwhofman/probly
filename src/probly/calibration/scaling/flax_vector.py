"""Implementation for Vector Scaling in Flax."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp

from probly.calibration.scaling import common

from .flax_base import ScalerFlax


class FlaxVector(ScalerFlax):
    """Vector Scaling Implementation with Flax."""

    def __init__(self, base: nnx.Module, num_classes: int) -> None:
        """Initialize Wrapper with w and biases.

        Args:
            base: The base model that should be calibrated.
            num_classes: The number of classes the base model was trained on (expects > 1).
        """
        if num_classes <= 1:
            msg = "vector scaling expects num_classes > 1."
            raise ValueError(msg)

        super().__init__(base, num_classes)

        self.w = nnx.Param(jnp.ones((num_classes,), dtype=jnp.float32))
        self.b = nnx.Param(jnp.zeros((num_classes,), dtype=jnp.float32))

    def _scale_logits(self, logits: jax.Array) -> jax.Array:
        """Scale logits based on learned parameters."""
        if logits.ndim < 2 or logits.shape[-1] != self.num_classes:
            msg = "vector scaling expects logits shape (...,K) with K=num_classes."
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
        if logits.ndim < 2 or logits.shape[-1] != self.num_classes:
            msg = "vector scaling expects logits shape (...,K) with K=num_classes."
            raise ValueError(msg)

        w = params["w"]
        b = params["b"]

        z = logits * w + b
        z = z.reshape((-1, z.shape[-1]))
        labels = labels.astype(jnp.int32).reshape((-1,))

        log_probs = jax.nn.log_softmax(z, axis=-1)
        nll = -log_probs[jnp.arange(labels.shape[0]), labels]

        return jnp.mean(nll)


@common.register_vector_factory(nnx.Module)
def _(_base: nnx.Module, _num_classes: int) -> type[FlaxVector]:
    return FlaxVector
