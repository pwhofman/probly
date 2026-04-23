"""Small plotting helpers shared across experiments."""

from __future__ import annotations

import numpy as np


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Return a centred rolling mean with edge handling via numpy convolution.

    Args:
        x: 1-D array to smooth.
        window: Window width in samples.

    Returns:
        A float array of the same length as ``x``. If ``len(x) < window`` the
        input is returned unchanged (cast to float).
    """
    if len(x) < window:
        return x.astype(float)
    kernel = np.ones(window) / window
    return np.convolve(x.astype(float), kernel, mode="same")
