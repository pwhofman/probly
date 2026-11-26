"""Array operations for the measures of numpy-based credal sets."""

from __future__ import annotations

import numpy as np

from ._common import _constant_like, _take_along_axis


@_take_along_axis.register(np.ndarray)
def _numpy_take_along_axis(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Gather the entries at the indices along the last axis."""
    return np.take_along_axis(values, indices, axis=-1)


@_constant_like.register(np.ndarray)
def _numpy_constant_like(reference: np.ndarray, values: object) -> np.ndarray:
    """Create a NumPy array with the dtype of the reference."""
    return np.array(values, dtype=reference.dtype)
