"""NumPy implementation of ROC AUC score."""

from __future__ import annotations

import numpy as np

from ._common import auc, roc_auc_score, roc_curve


@roc_auc_score.register(np.ndarray)
def roc_auc_score_numpy(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """Compute area under the ROC curve for NumPy arrays."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)
