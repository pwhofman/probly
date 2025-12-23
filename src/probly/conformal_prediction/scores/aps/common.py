from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

import numpy as np

from lazy_dispatch import lazydispatch
from lazy_dispatch.isinstance import LazyType

from ..common import Score


@lazydispatch
def aps_score_func(probs: Any) -> Any:
    """Compute APS scores for numpy arrays."""
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    cumsum_probs = np.cumsum(sorted_probs, axis=1)
    ranks = np.arange(1, probs.shape[1] + 1)
    aps_scores = np.sum(cumsum_probs / ranks, axis=1)
    return aps_scores


def register(cls: LazyType, func: Callable) -> None:
    """Register a class which can be used for APS score computation."""
    aps_score_func.register(cls=cls, func=func)


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

    return aps_score_func(probs)


def calculate_nonconformity_score(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute true-label APS nonconformity scores."""
    all_scores = aps_scores_all_labels(probabilities)
    labels = np.asarray(labels, dtype=int)
    n = labels.shape[0]
    return aps_score_func(probs)[np.arange(len(labels)), labels]


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
        probs = self.model.predict(x_cal)
        all_scores = aps_score_func(probs)

        # convert to NumPy
        if not isinstance(all_scores, np.ndarray):
            all_scores = np.asarray(all_scores)

        y_array = np.asarray(y_cal, dtype=int)
        n = y_array.shape[0]
        return calculate_nonconformity_score(probs, labels)

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
    ) -> np.ndarray:
        """Compute APS scores for all labels on test data.

        Returns a (n_samples, n_classes) score matrix that can be
        thresholded by the conformal predictor to build prediction sets.
        """
        probs = self.model.predict(x_test)
        all_scores = aps_score_func(probs)

        # convert to NumPy
        if not isinstance(all_scores, np.ndarray):
            all_scores = np.asarray(all_scores)

        return aps_score_func(probs)
