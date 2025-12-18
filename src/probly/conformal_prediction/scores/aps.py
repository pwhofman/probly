"""APS score implementation for conformal prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from probly.conformal_prediction.methods.common import PredictiveModel

import numpy as np

from .common import Score


def aps_scores_all_labels(
    probabilities: np.ndarray,
) -> np.ndarray:
    """Compute APS scores for all labels using cumulative probabilities."""
    probs = np.asarray(probabilities, dtype=float)

    # sort indices for descending probabilities
    srt_idx = np.argsort(-probs, axis=1)
    # sorted (negative) probabilities in descending order
    srt_probs = np.sort(-probs, axis=1)
    csum = -srt_probs.cumsum(axis=1)  # cumulative sum of original probs

    # scatter cumulative sums back to original label positions
    scores = np.zeros_like(probs, dtype=float)
    np.put_along_axis(scores, srt_idx, csum, axis=1)

    return scores


def calculate_nonconformity_score(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute true-label APS nonconformity scores."""
    all_scores = aps_scores_all_labels(probabilities)
    labels = np.asarray(labels, dtype=int)
    n = labels.shape[0]
    return all_scores[np.arange(n), labels]


def create_aps_prediction_sets(
    probs: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Build APS prediction sets as a binary mask.

    For each sample, classes are added in order of probability until
    the cumulative probability passes the threshold. At least one
    class is always selected.
    """
    probs = np.asarray(probs, dtype=float)

    # sort indices per sample in descending probability order
    sorted_idx = np.argsort(probs, axis=1)[:, ::-1]
    sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)

    # boolean mask: included while cumulative <= threshold
    include_mask = cumsum <= threshold

    # ensure at least one label per row
    all_false = ~include_mask.any(axis=1)
    if np.any(all_false):
        # force the top-probability label to be included
        include_mask[all_false, 0] = True

    # scatter mask back to original class indices
    prediction_sets = np.zeros_like(include_mask, dtype=bool)
    np.put_along_axis(prediction_sets, sorted_idx, include_mask, axis=1)

    return prediction_sets


class APSScore(Score):
    """APS nonconformity score based on model probabilities.

    The wrapped model is expected to implement
    predict(x: Sequence[Any]) -> np.ndarray of shape (n_samples, n_classes)
    returning class probabilities.
    """

    def __init__(self, model: PredictiveModel) -> None:
        """Initialize the APS score with a predictive model."""
        self.model = model

    def calibration_nonconformity(
        self,
        x_cal: Sequence[Any],
        y_cal: Sequence[Any],
    ) -> np.ndarray:
        """Compute true-label calibration scores."""
        probabilities = self.model.predict(x_cal)
        y_array = np.asarray(y_cal, dtype=int)
        return calculate_nonconformity_score(probabilities, y_array)

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
    ) -> np.ndarray:
        """Compute APS scores for all labels on test data.

        Returns a (n_samples, n_classes) score matrix that can be
        thresholded by the conformal predictor to build prediction sets.
        """
        probabilities = self.model.predict(x_test)
        return aps_scores_all_labels(probabilities)
