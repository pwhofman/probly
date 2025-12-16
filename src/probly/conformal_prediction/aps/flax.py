"""Flax Implementation of APS (Adaptive Prediction Sets)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from flax import nnx

    class CallableModule(nnx.Module):
        """Callable Flax module type hint."""

        def __call__(self, x: jax.Array) -> jax.Array:
            """Forward pass of the module."""
            ...


import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from probly.conformal_prediction.aps.common import calculate_nonconformity_score
from probly.conformal_prediction.common import ConformalPredictor


class FlaxAPS(ConformalPredictor):
    """Flax implementation of APS (Adaptive Prediction Sets)."""

    def __init__(
        self,
        model: nnx.Module,
        rng_key: jax.Array | int | None = None,
    ) -> None:
        """Initialize Flax APS predictor."""

        # Create wrapper for PredictiveModel protocol
        class FlaxModelWrapper:
            def __init__(self, flax_nnx_model: nnx.Module) -> None:
                self.flax_model = flax_nnx_model

            def predict(self, x: Sequence[Any]) -> npt.NDArray[np.floating]:
                """Convert input to probabilities."""
                # Ensure input is float32 for JAX
                x_array = jnp.asarray(x, dtype=jnp.float32)
                model_callable = cast("Callable[[jax.Array], jax.Array]", self.flax_model)
                logits = model_callable(x_array)
                probs = jax.nn.softmax(logits, axis=-1)
                # Return as float32 numpy array
                return np.asarray(probs, dtype=np.float32)

        # Initialize base class with wrapper
        super().__init__(model=FlaxModelWrapper(model), nonconformity_func=None)

        # Store Flax model
        self.flax_model = cast("Callable[[jax.Array], jax.Array]", model)

        # Handle random key
        if isinstance(rng_key, int):
            self.rng = jax.random.PRNGKey(rng_key)
        elif rng_key is not None:
            self.rng = rng_key
        else:
            self.rng = jax.random.PRNGKey(42)

    def _compute_nonconformity(self, x: Sequence[Any], y: Sequence[Any]) -> npt.NDArray[np.floating]:
        """Compute nonconformity scores using the APS method.

        Args:
            x: Input features (Sequence[Any] per API contract)
            y: True labels (Sequence[Any] per API contract)

        Returns:
            Nonconformity scores as float32
        """
        # Get probabilities from wrapped model (already float32)
        probabilities = self.model.predict(x)

        # Convert labels to numpy int32
        y_array = np.asarray(y, dtype=np.int32)

        # Use the common APS function
        scores = calculate_nonconformity_score(probabilities, y_array)

        # Ensure float32 and clip for numerical stability
        scores_float32 = np.asarray(scores, dtype=np.float32)
        return np.clip(scores_float32, 0.0, 1.0)  # APS scores should be in [0, 1]

    def predict(self, x: Sequence[Any], _significance_level: float) -> list[set[int]]:
        """Generate adaptive prediction sets.

        Args:
            x: Input features
            significance_level: Desired significance (e.g., 0.1 for 90% coverage)

        Returns:
            List of prediction sets (each set contains class indices)
        """
        if not self.is_calibrated or self.threshold is None:
            msg = "Model is not calibrated. Call calibrate() before predict()."
            raise ValueError(msg)

        # Convert input to JAX array (float32)
        x_jax = jnp.asarray(x, dtype=jnp.float32)

        # Get probabilities
        logits = self.flax_model(x_jax)
        probs = jax.nn.softmax(logits, axis=-1)
        probs_np = np.asarray(probs, dtype=np.float32)

        # Create prediction sets
        return self._create_prediction_sets(probs_np, self.threshold)

    def _create_prediction_sets(
        self,
        probs: npt.NDArray[np.floating],
        threshold: float,
    ) -> list[set[int]]:
        """Create adaptive prediction sets from probabilities."""
        n_samples = probs.shape[0]
        prediction_sets = []

        for i in range(n_samples):
            sample_probs = probs[i]

            # Sort indices by probability (descending)
            sorted_indices = np.argsort(sample_probs)[::-1]
            sorted_probs = sample_probs[sorted_indices]

            # Include classes until cumulative probability > threshold
            current_set = set()
            cumulative = 0.0

            for idx, prob in zip(sorted_indices, sorted_probs, strict=False):
                current_set.add(int(idx))
                cumulative += prob
                if cumulative > threshold:
                    break

            # Always include at least one class
            if not current_set:
                current_set.add(int(sorted_indices[0]))

            prediction_sets.append(current_set)

        return prediction_sets

    def jit_predict(self, x: jnp.ndarray) -> jnp.ndarray:
        """JIT-compiled prediction for performance.

        Args:
            x: Input features as JAX array

        Returns:
            Probabilities as JAX array
        """

        @jax.jit
        def predict_fn(x_input: jnp.ndarray) -> jnp.ndarray:
            logits = self.flax_model(x_input)
            return jax.nn.softmax(logits, axis=-1)

        return cast("jnp.ndarray", predict_fn(x))

    def __str__(self) -> str:
        """String representation."""
        model_name = self.flax_model.__class__.__name__
        status = "calibrated" if self.is_calibrated else "not calibrated"
        return f"FlaxAPS(model={model_name}, status={status})"
