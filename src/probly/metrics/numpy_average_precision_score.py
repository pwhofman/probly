"""NumPy implementation of average precision score."""

from __future__ import annotations

import numpy as np

from ._common import average_precision_score, precision_recall_curve


@average_precision_score.register(np.ndarray)
def average_precision_score_numpy(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """Compute average precision for NumPy arrays."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return -np.sum(np.diff(recall, axis=-1) * precision[..., :-1], axis=-1)
