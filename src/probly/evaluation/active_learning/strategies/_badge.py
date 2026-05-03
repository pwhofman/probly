"""BADGE (Batch Active learning by Diverse Gradient Embeddings) strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

from flextype import flexdispatch

from ._protocols import BadgeEstimator, Estimator

if TYPE_CHECKING:
    from probly.evaluation.active_learning.pool import ActiveLearningPool
    from probly.representation.array_like import ArrayLike


@flexdispatch
def badge_embed(estimator: object, x_unlabeled: ArrayLike) -> ArrayLike:
    """Extract embeddings for BADGE selection.

    Dispatches on the estimator type. If the estimator implements
    :class:`BadgeEstimator`, uses its ``embed`` method. Otherwise, falls back
    to using ``x_unlabeled`` directly with a warning.

    Args:
        estimator: A fitted estimator.
        x_unlabeled: The unlabeled feature matrix (used as fallback embeddings).

    Returns:
        Embeddings of shape (n_pool, emb_dim).
    """
    warnings.warn(
        f"BADGEQuery received estimator of type {type(estimator).__name__} which "
        "does not implement BadgeEstimator (no .embed() method). Falling back to "
        "pool.x_unlabeled as embeddings; BADGE was designed for penultimate-layer "
        "features and selection quality may degrade.",
        UserWarning,
        stacklevel=3,
    )
    return x_unlabeled


@badge_embed.register(BadgeEstimator)
def _badge_embed_badge(estimator: BadgeEstimator, x_unlabeled: ArrayLike) -> ArrayLike:
    """Use the estimator's penultimate-layer embeddings for BADGE selection."""
    return estimator.embed(x_unlabeled)


@flexdispatch
def badge_select(
    embeddings: ArrayLike,
    probs: ArrayLike,
    n: int,
    seed: int | None = None,
) -> ArrayLike:
    """Select n indices via BADGE gradient embedding k-means++.

    Args:
        embeddings: Feature embeddings of shape (n_pool, emb_dim).
        probs: Predicted class probabilities of shape (n_pool, n_classes).
        n: Number of indices to select.
        seed: Seed for the random number generator.

    Returns:
        Array of n integer indices.
    """
    msg = f"No badge_select implementation registered for type {type(embeddings)}"
    raise NotImplementedError(msg)


class BADGEQuery:
    """Selects a diverse uncertain batch via BADGE gradient embedding k-means++.

    Implements Batch Active learning by Diverse Gradient Embeddings (Ash et al.,
    2020). If the estimator implements :class:`BadgeEstimator` (has an
    ``embed`` method) it is used for embeddings; otherwise ``pool.x_unlabeled``
    is used directly and a :class:`UserWarning` is emitted.

    Args:
        seed: Seed for the k-means++ initialization. Pass None for
            non-deterministic selection.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Store the seed used for k-means++ initialization.

        Args:
            seed: Seed for the random number generator. Pass None for
                non-deterministic selection.
        """
        self._seed = seed

    def select(self, estimator: Estimator, pool: ActiveLearningPool, n: int) -> ArrayLike:
        """Return n indices chosen by BADGE k-means++ on gradient embeddings."""
        n = min(n, pool.n_unlabeled)
        probs = estimator.predict_proba(pool.x_unlabeled)
        embeddings = badge_embed(estimator, pool.x_unlabeled)
        return badge_select(embeddings, probs, n, seed=self._seed)
