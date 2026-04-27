"""Protocols, dispatched select functions, and strategy classes for active learning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable
import warnings

import numpy as np

from flextype import flexdispatch

if TYPE_CHECKING:
    from probly.evaluation.active_learning.pool import ActiveLearningPool
    from probly.representation.array_like import ArrayLike


@runtime_checkable
class Estimator(Protocol):
    """Protocol for estimators usable by query strategies."""

    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        """Fit the estimator on labeled data."""
        ...

    def predict(self, x: ArrayLike) -> ArrayLike:
        """Return class predictions."""
        ...

    def predict_proba(self, x: ArrayLike) -> ArrayLike:
        """Return class probabilities."""
        ...


@runtime_checkable
class UncertaintyEstimator(Estimator, Protocol):
    """Protocol for estimators that provide uncertainty scores."""

    def uncertainty_scores(self, x: ArrayLike) -> ArrayLike:
        """Return per-sample uncertainty scores of shape (n_samples,)."""
        ...


@runtime_checkable
class BadgeEstimator(Estimator, Protocol):
    """Protocol for estimators usable by :class:`BADGEQuery`.

    Extends :class:`Estimator` with an ``embed`` method that returns
    penultimate-layer features used to build BADGE gradient embeddings.
    """

    def embed(self, x: ArrayLike) -> ArrayLike:
        """Return penultimate-layer embeddings of shape ``(n, emb_dim)``."""
        ...


class QueryStrategy[E: Estimator](Protocol):
    """Protocol for active learning query strategies."""

    def select(self, estimator: E, pool: ActiveLearningPool, n: int) -> ArrayLike:
        """Return n indices into pool.x_unlabeled to query next.

        Args:
            estimator: A fitted estimator used to score unlabeled samples.
            pool: The current active learning pool.
            n: Number of samples to select.

        Returns:
            Array of n integer indices into pool.x_unlabeled.
        """
        ...


# ---------------------------------------------------------------------------
# Dispatched select functions
# ---------------------------------------------------------------------------


@flexdispatch
def margin_select(probs: ArrayLike, n: int) -> ArrayLike:
    """Select n indices with the smallest margin between top-2 class probabilities.

    Args:
        probs: Class probability matrix of shape (n_pool, n_classes).
        n: Number of indices to select.

    Returns:
        Array of n integer indices.
    """
    msg = f"No margin_select implementation registered for type {type(probs)}"
    raise NotImplementedError(msg)


@flexdispatch
def uncertainty_select(scores: ArrayLike, n: int) -> ArrayLike:
    """Select n indices with the highest uncertainty scores.

    Args:
        scores: Per-sample uncertainty scores of shape (n_pool,).
        n: Number of indices to select.

    Returns:
        Array of n integer indices.
    """
    msg = f"No uncertainty_select implementation registered for type {type(scores)}"
    raise NotImplementedError(msg)


@flexdispatch
def badge_embed(estimator: object, x_unlabeled: ArrayLike) -> ArrayLike:
    """Extract embeddings for BADGE selection.

    Dispatches on the estimator type. If the estimator implements
    :class:`BadgeEstimator`, uses its ``embed`` method. Otherwise falls back
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


# ---------------------------------------------------------------------------
# Strategy classes (thin wrappers around dispatched functions)
# ---------------------------------------------------------------------------


class RandomQuery:
    """Selects unlabeled samples uniformly at random.

    Args:
        seed: Seed for the random number generator. Pass None for
            non-deterministic selection.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the random number generator.

        Args:
            seed: Seed for the random number generator. Pass None for
                non-deterministic selection.
        """
        self._rng = np.random.default_rng(seed)

    def select(
        self,
        estimator: Estimator,  # noqa: ARG002
        pool: ActiveLearningPool,
        n: int,
    ) -> ArrayLike:
        """Return n randomly chosen indices from the unlabeled pool.

        Args:
            estimator: Fitted estimator (not used by this strategy).
            pool: The current active learning pool.
            n: Number of samples to select.

        Returns:
            Array of n unique integer indices into pool.x_unlabeled.
        """
        n = min(n, pool.n_unlabeled)
        return random_select(pool.x_unlabeled, pool.n_unlabeled, n, self._rng)


class MarginSampling:
    """Selects unlabeled samples with the smallest margin between top-2 class probs.

    A small margin indicates the model is uncertain between its top two predicted
    classes, making these samples most informative to label.
    """

    def select(self, estimator: Estimator, pool: ActiveLearningPool, n: int) -> ArrayLike:
        """Return n indices with the smallest margin in predicted probabilities.

        Args:
            estimator: Fitted estimator with a predict_proba method.
            pool: The current active learning pool.
            n: Number of samples to select.

        Returns:
            Array of n integer indices into pool.x_unlabeled.
        """
        n = min(n, pool.n_unlabeled)
        probs = estimator.predict_proba(pool.x_unlabeled)
        return margin_select(probs, n)


class UncertaintyQuery:
    """Selects samples with the highest estimator-provided uncertainty scores.

    Delegates scoring to estimator.uncertainty_scores(). Suitable for any probly
    UQ method that provides its own per-sample scoring (e.g. mutual information
    or any other UQ-based measure).
    """

    def select(self, estimator: UncertaintyEstimator, pool: ActiveLearningPool, n: int) -> ArrayLike:
        """Return n indices with the highest uncertainty scores from the estimator.

        Args:
            estimator: Fitted estimator with an uncertainty_scores method.
            pool: The current active learning pool.
            n: Number of samples to select.

        Returns:
            Array of n integer indices into pool.x_unlabeled.
        """
        n = min(n, pool.n_unlabeled)
        scores = estimator.uncertainty_scores(pool.x_unlabeled)
        return uncertainty_select(scores, n)


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
        """Return n indices chosen by BADGE k-means++ on gradient embeddings.

        Args:
            estimator: Fitted estimator with predict_proba; optionally embed().
            pool: The current active learning pool.
            n: Number of samples to select.

        Returns:
            Array of n integer indices into pool.x_unlabeled.
        """
        n = min(n, pool.n_unlabeled)
        probs = estimator.predict_proba(pool.x_unlabeled)
        embeddings = badge_embed(estimator, pool.x_unlabeled)
        return badge_select(embeddings, probs, n, seed=self._seed)
