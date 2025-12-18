"""FlaxNNXWrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from flax import nnx

import jax
from jax import Array
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from probly.conformal_prediction.methods.common import PredictiveModel


class FlaxNNXWrapper(PredictiveModel):
    """Wrapper for flax nnx.Module and callable flax model."""

    def __init__(self, flax_nnx_model: nnx.Module) -> None:
        """Initialize Wrapper with a nnx.Module or callable model.

        Args:
            flax_nnx_model: nnx.Module or callable model that maps
            input arrays to output logits.
        """
        self.flax_model = flax_nnx_model

    def predict(self, x: Sequence[Any]) -> npt.NDArray[np.floating]:
        """Calculate probabilities as numpy arrays."""
        x_array = jnp.asarray(x, dtype=jnp.float32)
        model_callable = cast("Callable[[jax.Array], jax.Array]", self.flax_model)
        logits = model_callable(x_array)
        probs = jax.nn.softmax(logits, axis=-1)
        return np.asarray(probs, dtype=np.float32)

    def jit_predict(self, x: jnp.ndarray) -> jnp.ndarray:
        """Jit compiled prediction."""
        model_callable = cast("Callable[[Array], Array]", self.flax_model)

        @jax.jit
        def predict_fn(x_input: jnp.ndarray) -> jnp.ndarray:
            logits = model_callable(x_input)
            return jax.nn.softmax(logits, axis=-1)

        return cast("jnp.ndarray", predict_fn(x))
