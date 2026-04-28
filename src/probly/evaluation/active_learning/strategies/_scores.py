"""Dispatched scoring functions for active learning strategies.

All scoring functions follow the convention: higher score = more informative sample.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flextype import flexdispatch

if TYPE_CHECKING:
    from probly.representation.array_like import ArrayLike


@flexdispatch
def entropy_score(probs: ArrayLike) -> ArrayLike:
    """Score each sample by prediction entropy. Higher entropy = more informative.

    Args:
        probs: Class probability matrix of shape (n_pool, n_classes).

    Returns:
        Per-sample entropy scores of shape (n_pool,).
    """
    msg = f"No entropy_score implementation registered for type {type(probs)}"
    raise NotImplementedError(msg)


@flexdispatch
def least_confident_score(probs: ArrayLike) -> ArrayLike:
    """Score each sample by 1 - max(prob). Higher = less confident = more informative.

    Args:
        probs: Class probability matrix of shape (n_pool, n_classes).

    Returns:
        Per-sample scores of shape (n_pool,).
    """
    msg = f"No least_confident_score implementation registered for type {type(probs)}"
    raise NotImplementedError(msg)


@flexdispatch
def margin_score(probs: ArrayLike) -> ArrayLike:
    """Score each sample by negative margin. Higher = smaller margin = more informative.

    Args:
        probs: Class probability matrix of shape (n_pool, n_classes).

    Returns:
        Per-sample scores of shape (n_pool,). Negative margin so that higher = more informative.
    """
    msg = f"No margin_score implementation registered for type {type(probs)}"
    raise NotImplementedError(msg)
