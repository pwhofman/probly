"""Utility functions for NumPy arrays."""

from __future__ import annotations

import numpy as np

from ._common import elementwise_entropy, entropy


@elementwise_entropy.register(np.ndarray)
def numpy_elementwise_entropy(p: np.ndarray) -> np.ndarray:
    """Entropy contributions ``-p * log(p)`` computed in numpy; 0*log(0) treated as 0.

    Args:
        p: Probabilities to compute the entropy contributions of.

    Returns:
        Entropy contributions of the shape of ``p``.
    """
    safe_p = np.where(p > 0, p, np.ones_like(p))
    return -(p * np.log(safe_p))


@entropy.register(np.ndarray)
def numpy_entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy H(p) computed in numpy along the last axis; 0*log(0) treated as 0.

    Args:
        p: Probabilities to compute entropy of.

    Returns:
        Entropy of probabilities p.
    """
    result = numpy_elementwise_entropy(p).sum(axis=-1)
    return np.clip(result, 0.0, None) + 0.0  # Ensure non-negativity
