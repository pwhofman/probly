"""Protocols, dispatched select functions, and strategy classes for active learning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable
import warnings

import numpy as np

from flextype import flexdispatch

if TYPE_CHECKING:
    from probly.evaluation.active_learning.pool._common import ActiveLearningPool
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
class BadgeEstimator(Estimator, Protocol):
    """Protocol for estimators usable by :class:`BADGEQuery`.

    Extends :class:`Estimator` with an ``embed`` method that returns
    penultimate-layer features used to build BADGE gradient embeddings.
    """

    def embed(self, x: ArrayLike) -> ArrayLike:
        """Return penultimate-layer embeddings of shape ``(n, emb_dim)``."""
        ...


class QueryStrategy(Protocol):
    """Protocol for active learning query strategies."""

    def select(self, estimator: Estimator, pool: ActiveLearningPool, n: int) -> np.ndarray:
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
def margin_select(probs: object, n: int) -> np.ndarray:
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
def uncertainty_select(scores: object, n: int) -> np.ndarray:
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
def badge_select(
    embeddings: object,
    probs: object,
    n: int,
    seed: int | None = None,
) -> np.ndarray:
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
        self._rng = np.random.default_rng(seed)

    def select(self, estimator: Estimator, pool: ActiveLearningPool, n: int) -> np.ndarray:  # noqa: ARG002
        """Return n randomly chosen indices from the unlabeled pool.

        Args:
            estimator: Fitted estimator (not used by this strategy).
            pool: The current active learning pool.
            n: Number of samples to select.

        Returns:
            Array of n unique integer indices into pool.x_unlabeled.
        """
        n = min(n, pool.n_unlabeled)
        return self._rng.choice(pool.n_unlabeled, size=n, replace=False)


class MarginSampling:
    """Selects unlabeled samples with the smallest margin between top-2 class probs.

    A small margin indicates the model is uncertain between its top two predicted
    classes, making these samples most informative to label.
    """

    def select(self, estimator: Estimator, pool: ActiveLearningPool, n: int) -> np.ndarray:
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

    def select(self, estimator: Estimator, pool: ActiveLearningPool, n: int) -> np.ndarray:
        """Return n indices with the highest uncertainty scores from the estimator.

        Args:
            estimator: Fitted estimator with an uncertainty_scores method.
            pool: The current active learning pool.
            n: Number of samples to select.

        Returns:
            Array of n integer indices into pool.x_unlabeled.
        """
        n = min(n, pool.n_unlabeled)
        scores = cast("Any", estimator).uncertainty_scores(pool.x_unlabeled)
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
        self._seed = seed

    def select(self, estimator: Estimator, pool: ActiveLearningPool, n: int) -> np.ndarray:
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
        if isinstance(estimator, BadgeEstimator):
            embeddings = estimator.embed(pool.x_unlabeled)
        else:
            warnings.warn(
                f"BADGEQuery received estimator of type {type(estimator).__name__} which "
                "does not implement BadgeEstimator (no .embed() method). Falling back to "
                "pool.x_unlabeled as embeddings; BADGE was designed for penultimate-layer "
                "features and selection quality may degrade.",
                UserWarning,
                stacklevel=2,
            )
            embeddings = pool.x_unlabeled
        return badge_select(embeddings, probs, n, seed=self._seed)
