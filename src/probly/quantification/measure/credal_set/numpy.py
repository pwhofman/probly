"""Array operations for the measures of numpy-based credal sets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._common import _constant_like, _take_along_axis, _where

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


@_take_along_axis.register(np.ndarray)
def _numpy_take_along_axis(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Gather the entries at the indices along the last axis."""
    indices = np.broadcast_to(indices, (*values.shape[:-1], indices.shape[-1]))
    return np.take_along_axis(values, indices, axis=-1)


@_constant_like.register(np.ndarray)
def _numpy_constant_like(reference: np.ndarray, values: object) -> np.ndarray:
    """Create a NumPy array with the dtype of the reference."""
    return np.array(values, dtype=reference.dtype)


# Comparisons of unbatched arrays give NumPy scalars instead of arrays.
@_where.register((np.ndarray, np.generic))
def _numpy_where(condition: np.ndarray | np.generic, a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Choose between two arrays by a boolean condition."""
    return np.where(condition, a, b)
