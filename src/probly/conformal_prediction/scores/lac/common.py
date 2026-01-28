"""Common functions for LAC Nonconformity-Scores."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from lazy_dispatch.isinstance import LazyType
    from probly.conformal_prediction.methods.common import Predictor

import numpy as np
import numpy.typing as npt

from lazy_dispatch import lazydispatch
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
        super().__init__(model=model, score_func=lac_score_func)
