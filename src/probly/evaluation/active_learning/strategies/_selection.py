"""Dispatched selection functions for active learning strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flextype import flexdispatch

if TYPE_CHECKING:
    import numpy as np

    from probly.representation.array_like import ArrayLike


@flexdispatch
def topk_select(scores: ArrayLike, n: int) -> ArrayLike:
    """Select n indices with the highest scores.

    Args:
        scores: Per-sample scores of shape (n_pool,). Higher = more informative.
        n: Number of indices to select.

    Returns:
        Array of n integer indices.
    """
    msg = f"No topk_select implementation registered for type {type(scores)}"
    raise NotImplementedError(msg)


@flexdispatch
def random_select(x_ref: ArrayLike, n_pool: int, n: int, rng: np.random.Generator) -> ArrayLike:
    """Select n unique random indices, returning the backend's native index type.

    Dispatches on the type of ``x_ref`` (a reference array from the pool, used
    only for backend detection).

    Args:
        x_ref: Reference array for backend dispatch (e.g. ``pool.x_unlabeled``).
        n_pool: Total number of items to choose from.
        n: Number of indices to select.
        rng: A ``numpy.random.Generator`` instance for reproducible sampling.

    Returns:
        Array of n unique integer indices in the backend's native type.
    """
    msg = f"No random_select implementation registered for type {type(x_ref)}"
    raise NotImplementedError(msg)
