from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

class DropConnectLinear(nnx.Module):
    """
    Linear mit DropConnect auf die Gewichte während des Trainings.
    Wrappt eine bestehende nnx.Linear-Schicht.
    """
    def __init__(self, base_layer: nnx.Linear, p: float = 0.25) -> None:
        if not (0.0 <= p < 1.0):
            raise ValueError(f"p must be in [0, 1); got {p}")
        self.base_layer = base_layer
        self.p = float(p)

        # Reproduzierbarer RNG-State (einfach gehalten)
        self._key = jax.random.PRNGKey(0)

        # Metadaten aus der Basisschicht – robust auf weight/kernel
        self.in_features = getattr(base_layer, "in_features", None)
        self.out_features = getattr(base_layer, "out_features", None)

    def __call__(self, x: jnp.ndarray, *, training: bool | None = None) -> jnp.ndarray:
        """
        Forward-Pass mit DropConnect.
        Args:
            x: [batch, in_features]
            training: Wenn True → DropConnect aktiv; wenn False → Inference-Skalierung;
                      wenn None → fallback auf getattr(self, 'training', False)
        """
        # Gewicht/Bias robust auslesen (NNX vs linen)
        weight = getattr(self.base_layer, "weight", None)
        if weight is None:
            weight = getattr(self.base_layer, "kernel")  # linen-Kompatibilität
        bias = getattr(self.base_layer, "bias", None)

        if training is None:
            training = getattr(self, "training", False)

        if training:
            self._key, subkey = jax.random.split(self._key)
            keep_prob = 1.0 - self.p
            mask = jax.random.bernoulli(subkey, p=keep_prob, shape=weight.shape)
            # dtype an Gewicht anpassen (wichtig bei float16/bfloat16)
            mask = mask.astype(weight.dtype)
            eff_weight = weight * mask
        else:
            # Erwartungswert-Korrektur bei Inference
            eff_weight = weight * (1.0 - self.p)

        # Linearer Vorwärtslauf
        # (bewusst eigene MatMul statt base_layer(x), da wir eff_weight verwenden)
        y = jnp.matmul(x, eff_weight)
        if bias is not None:
            y = y + bias
        return y

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={getattr(self.base_layer, 'bias', None) is not None}, "
            f"p={self.p}"
        )