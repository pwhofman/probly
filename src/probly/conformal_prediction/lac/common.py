"""Shared implementation for LAC (Least-ambigious-classifier).

Implementation of Local Aggregative Conformal (LAC) prediction.
Contains the LAC class and all necessary helper methods in one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

# Import the global base class
from probly.conformal_prediction.common import ConformalPredictor

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

# --- HELPER FUNCTIONS ---


def calculate_non_conformity_score(
    probas: npt.NDArray[np.floating],
    y_indices: npt.NDArray[np.integer],
) -> npt.NDArray[np.floating]:
    """Compute Non-Conformity Scores for LAC.

    Score calculation: s(x, y) = 1 - p(y|x).
    Fulfills goal: calculate.non-comformityscore()
    """
    n_samples = len(y_indices)
    # Extract the probability of the true class
    # Advanced Indexing: [0...n, y_true]
    true_class_probas = probas[np.arange(n_samples), y_indices]

    # Calculate score as 1 minus probability
    scores = 1.0 - true_class_probas
    return scores


def calculate_local_weights(
    x: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Calculates local weights. For standard LAC Split, returns uniform weights.

    Fulfills goal: calculate.local_weights()
    """
    n_samples = x.shape[0]
    return np.ones(n_samples, dtype=float)


def calculate_weighted_quantile(
    values: npt.NDArray[np.floating],
    quantile: float,
    sample_weight: npt.NDArray[np.floating] | None = None,
) -> float:
    """Calculates a weighted quantile of the values using numpy.

    Fulfills goal: calculate.weighted_quantile()
    """
    if sample_weight is None:
        return float(np.quantile(values, quantile, method="higher"))

    values = np.array(values)
    sample_weight = np.array(sample_weight)

    # Sort values and weights
    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    # Compute cumulative weights
    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_quantiles /= np.sum(sample_weight)

    # Interpolate
    return float(np.interp(quantile, weighted_quantiles, values))


def accretive_completion(
    prediction_sets: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.floating],
) -> npt.NDArray[np.bool_]:
    """Implements Accretive Completion to eliminate empty prediction sets (Null Regions).

    If a set is empty, it adds the class with the highest score (probability).
    """
    completed_sets = prediction_sets.copy()

    # Identify empty rows
    set_sizes = np.sum(completed_sets, axis=1)
    empty_rows_mask = set_sizes == 0

    if not np.any(empty_rows_mask):
        return completed_sets

    # For empty rows: Find index of the class with the highest probability
    # Note: 'scores' here are probabilities passed from predict()
    best_class_indices = np.argmax(scores[empty_rows_mask], axis=1)
    row_indices = np.where(empty_rows_mask)[0]

    # Force this class to True
    completed_sets[row_indices, best_class_indices] = True
    return completed_sets


# --- MAIN CLASS ---


class LAC(ConformalPredictor):
    """Least Ambiguous Set-Valued Classifier (LAC).

    Implements Split-Conformal Prediction with Accretive Completion.
    """

    def _compute_nonconformity(
        self,
        x: Sequence[Any],
        y: Sequence[Any],
    ) -> npt.NDArray[np.floating]:
        """Calculates non-conformity scores using 1 - p(y|x)."""
        # 1. Prediction (Probabilities)
        probas = self.model.predict(x)

        # 2. Prepare labels
        y_indices = np.asarray(y, dtype=int)

        # 3. Call helper function
        return calculate_non_conformity_score(probas, y_indices)

    def predict(
        self,
        x: Sequence[Any],
        significance_level: float,  # noqa: ARG002
    ) -> list[npt.NDArray[np.bool_]]:
        """Predicts sets and fixes empty ones using Accretive Completion."""
        if not self.is_calibrated or self.threshold is None:
            msg = "Predictor is not calibrated. Call calibrate() first."
            raise RuntimeError(msg)

        # 1. Get probabilities
        probas = self.model.predict(x)

        # 2. Convert threshold (since scores are 1-p)
        # Score <= Threshold <==> 1 - p <= Threshold <==> p >= 1 - Threshold
        prob_threshold = 1.0 - self.threshold

        # 3. Create initial sets
        # Class is included if probability >= prob_threshold
        prediction_sets = probas >= prob_threshold

        # 4. Apply Accretive Completion
        # Fixes empty sets (Null Regions) using the helper function above
        final_sets = accretive_completion(prediction_sets, probas)

        return list(final_sets)
