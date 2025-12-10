"""LAC implementation for Flax models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

# Import the general LAC class and helper functions
from probly.conformal_prediction.lac.common import (
    LAC,
    accretive_completion,
    calculate_non_conformity_score,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import flax.linen as nn
    import numpy.typing as npt


class FlaxModelWrapper:
    """Wrapper to make Flax models compatible with PredictiveModel protocol."""

    def __init__(self, flax_model: nn.Module, params: dict[str, Any]) -> None:
        """Initialize with a Flax model and its parameters."""
        self.flax_model = flax_model
        self.params = params

    def predict(self, x: Sequence[Any]) -> np.ndarray:
        """Predict method for PredictiveModel protocol."""
        x_jax = jnp.asarray(x)
        logits = self.flax_model.apply(self.params, x_jax)
        probas_jax = jax.nn.softmax(logits, axis=-1)
        return np.array(probas_jax, dtype=np.float32)


class LACFlax(LAC):
    """LAC Predictor specifically for Flax/JAX models.

    Wraps a Flax model to perform Least Ambiguous Set-Valued Classification.
    """

    def __init__(self, model: Any, params: Any) -> None:  # noqa: ANN401
        """Initialize with a Flax model and its parameters.

        Args:
            model: The Flax module (must have an 'apply' method).
            params: The frozen parameters (weights) of the model.
        """
        # Wrap the Flax model
        wrapped_model = FlaxModelWrapper(model, params)
        super().__init__(wrapped_model)
        self.params = params
        self.flax_model = model

    def _get_probabilities(self, x: npt.NDArray[Any]) -> npt.NDArray[np.floating]:
        """Internal method to get probabilities from the Flax model."""
        return self.model.predict(x)

    def predict(
        self,
        x: Sequence[Any],
        significance_level: float,  # noqa: ARG002
    ) -> list[npt.NDArray[np.bool_]]:
        """Predict sets using the Flax model.

        Overwrites the base predict to ensure we use JAX mechanics
        before calling the logic in common.py.
        """
        if not self.is_calibrated or self.threshold is None:
            msg = "Predictor is not calibrated. Call calibrate() first."
            raise RuntimeError(msg)

        # 1. Get probabilities using our custom JAX method
        # We assume x is compatible with numpy/jax conversion
        probas = self._get_probabilities(np.asarray(x))

        # 2. Convert threshold logic
        # Score <= Threshold <==> 1 - p <= Threshold <==> p >= 1 - Threshold
        prob_threshold = 1.0 - self.threshold

        # 3. Create initial sets
        prediction_sets = probas >= prob_threshold

        # 4. Apply Accretive Completion
        final_sets = accretive_completion(prediction_sets, probas)

        return list(final_sets)

    def _compute_nonconformity(
        self,
        x: Sequence[Any],
        y: Sequence[Any],
    ) -> npt.NDArray[np.floating]:
        """Compute non-conformity scores for Flax models."""
        # 1. Get probabilities
        probas = self._get_probabilities(np.asarray(x))

        # 2. Prepare labels
        y_indices = np.asarray(y, dtype=int)

        # 3. Call helper function from common.py
        return calculate_non_conformity_score(probas, y_indices)
