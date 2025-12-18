"""Common structures for conformal prediction scores."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


class Score(Protocol):
    """Interface for nonconformity scores used in split conformal prediction.

    Each score (APS, LAC, RAPS, ...) must implement:
    - calibration_nonconformity: used on calibration data.
    - predict_nonconformity: used on test data, must return a score matrix
      of shape (n_instances, n_labels).
    """

    def calibration_nonconformity(
        self,
        x_cal: Sequence[Any],
        y_cal: Sequence[Any],
    ) -> npt.NDArray[np.floating]:
        """Return 1D array of nonconformity scores for calibration instances."""

    def predict_nonconformity(
        self,
        x_test: Sequence[Any],
    ) -> npt.NDArray[np.floating]:
        """Return 2D score matrix of shape (n_instances, n_labels)."""


def calculate_quantile(scores: np.ndarray, alpha: float) -> float:
    """Calculate the quantile for conformal prediction.

    Parameters:
    scores : np.ndarray
            Non-conformity scores
    alpha : float
            Significance level (target coverage is 1-alpha)

    Returns:
    float (The (1-alpha)-quantile of the scores)
    """
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)  # ensure within [0, 1]
    return float(np.quantile(scores, q_level, method="inverted_cdf"))
