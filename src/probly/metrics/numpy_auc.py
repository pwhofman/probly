"""NumPy implementation of AUC."""

from __future__ import annotations

import numpy as np

from ._common import auc


@auc.register(np.ndarray)
def auc_numpy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute area under a curve using the trapezoid rule."""
    return np.trapezoid(y, x, axis=-1)
