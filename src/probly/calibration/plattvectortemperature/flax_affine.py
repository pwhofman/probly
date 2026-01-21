"""Implementation For Platt-  and Vector Scaling Extension of Base."""

from __future__ import annotations

from flax import nnx
import jax
import jax.numpy as jnp

from .flax_base import _ScalerFlax


class AffineScalingFlax(_ScalerFlax):
    """Wrapper Class for Platt- and Vectorscaling."""

    def __init__(self, base: nnx.Module, num_classes: int) -> None:
        """Initialize Wrapper with w and biases."""
        super().__init__(base, num_classes)

        self.w = nnx.Param(jnp.array((num_classes,), dtype=jnp.float32))
        self.b = nnx.Param(jnp.array((num_classes,), dtype=jnp.float32))

    def _scale_logits(self, logits: jax.Array) -> jax.Array:
        if self.num_classes == 1 and logits.ndim == 1:
            logits = logits[..., None]

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

        # Binary Case
        if self.num_classes == 1:
            if logits.ndim == 1:
                logits = logits[..., None]

            z = logits * w + b
            labels = labels.reshape(z.shape)

            loss = jnp.mean(jnp.maximum(z, 0) - z * labels + jnp.log1p(jnp.exp(-jnp.abs(z))))

            return loss

        # Multiclass Case
        z = logits * w + b
        labels = labels.astype(jnp.int32).reshape(-1)

        log_probs = z - jax.nn.logsumexp(z, axis=-1, keepdims=True)
        nll = -log_probs[jnp.arange(labels.shape[0]), labels]

        return jnp.mean(nll)
