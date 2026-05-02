"""Uncertainty estimation for RL agents."""

from __future__ import annotations

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
