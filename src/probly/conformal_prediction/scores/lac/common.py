"""Common functions for LAC Nonconformity-Scores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lazy_dispatch.isinstance import LazyType

import numpy as np
import numpy.typing as npt

from lazy_dispatch import lazydispatch
from probly.conformal_prediction.methods.common import Predictor, predict_probs
from probly.conformal_prediction.scores.common import ClassificationScore


@lazydispatch
def lac_score_func[T](probs: T) -> npt.NDArray[np.floating]:
    """LAC Nonconformity-Scores for numpy arrays."""
    # convert to numpy array
    probs_np = np.asarray(probs, dtype=float)
    lac_scores = 1.0 - probs_np
    return lac_scores  # shape: (n_samples, n_classes)


def register(cls: LazyType, func: Callable) -> None:
    """Register a class which can be used for LAC score computation."""
    lac_score_func.register(cls=cls, func=func)


def accretive_completion(
    prediction_sets: np.ndarray,
    probs: np.ndarray,
) -> np.ndarray:
    """Implements Accretive Completion to eliminate empty prediction sets (Null Regions).

    Args:
        prediction_sets (np.ndarray): Boolean array of shape (n_samples, n_classes).
                                      True indicates the class is in the set.
        probs (np.ndarray): Array of shape (n_samples, n_classes).
                           Usually conditional probabilities p(y|x).
                           High score implies higher likelihood of the class.

    Returns:
        np.ndarray: The modified prediction sets where every row has at least one True.
    """
    # ensure inputs are numpy arrays
    prediction_sets = np.asarray(prediction_sets)
    probs_np = np.asarray(probs)

    completed_sets = prediction_sets.copy()

    set_sizes = np.sum(completed_sets, axis=1)
    empty_rows_mask = set_sizes == 0

    if not np.any(empty_rows_mask):
        return completed_sets

    best_class_indx = np.argmax(probs_np[empty_rows_mask], axis=1)
    row_indx = np.where(empty_rows_mask)[0]

    completed_sets[row_indx, best_class_indx] = True
    return completed_sets


class LACScore(ClassificationScore):
    """LAC Nonconformity-Score."""

    def __init__(self, model: Predictor) -> None:
        """Initialize LAC score with model."""
        self.model = model

    def calibration_nonconformity(
        self,
        x_cal: Sequence[Any],
        y_cal: Sequence[Any],
        probs: npt.NDArray[np.floating] | None = None,
    ) -> npt.NDArray[np.floating]:
        """Compute true-label calibration scores."""
        # get probabilities from model
        if probs is None:
            probs = predict_probs(self.model, x_cal)
        # get lac scores for all labels
        all_scores: npt.NDArray[np.floating] = lac_score_func(probs)

        # convert to numpy arrays
        scores_np = np.asarray(all_scores, dtype=float)
        labels_np = np.asarray(y_cal, dtype=int)

        # extract scores for true labels
        idx = np.arange(len(labels_np))
        nonconformity: npt.NDArray[np.floating] = scores_np[idx, labels_np]

        return nonconformity

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
        probs: npt.NDArray[np.floating] | None = None,
    ) -> npt.NDArray[np.floating]:
        """Compute LAC scores for all labels on test data."""
        # predict
        if probs is None:
            probs = predict_probs(self.model, x_test)

        # compute all scores (1 - p)
        all_scores: npt.NDArray[np.floating] = lac_score_func(probs)  # shape: (n_samples, n_classes)
        scores_np = np.asarray(all_scores, dtype=float)

        return scores_np
