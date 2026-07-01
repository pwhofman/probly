"""NumPy implementations of scoring rule loss vectors."""

from __future__ import annotations

import numpy as np

from ._common import _brier_loss_vector, _log_loss_vector, _spherical_loss_vector, _zero_one_loss_vector


@_log_loss_vector.register
def array_log_loss_vector(probabilities: np.ndarray) -> np.ndarray:
    """Compute the per-label log loss vector for a NumPy array."""
    return -np.log(probabilities)


@_brier_loss_vector.register
def array_brier_loss_vector(probabilities: np.ndarray) -> np.ndarray:
    """Compute the per-label Brier loss vector for a NumPy array."""
    squared_norm = np.sum(probabilities**2, axis=-1, keepdims=True)
    return squared_norm - 2.0 * probabilities + 1.0


@_zero_one_loss_vector.register
def array_zero_one_loss_vector(probabilities: np.ndarray) -> np.ndarray:
    """Compute the per-label zero-one loss vector for a NumPy array."""
    num_classes = probabilities.shape[-1]
    argmax = np.argmax(probabilities, axis=-1)
    one_hot = np.eye(num_classes, dtype=probabilities.dtype)[argmax]
    return 1.0 - one_hot


@_spherical_loss_vector.register
def array_spherical_loss_vector(probabilities: np.ndarray) -> np.ndarray:
    """Compute the per-label spherical loss vector for a NumPy array."""
    norm = np.sqrt(np.sum(probabilities**2, axis=-1, keepdims=True))
    return 1.0 - probabilities / norm
