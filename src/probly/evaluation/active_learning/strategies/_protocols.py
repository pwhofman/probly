"""Estimator and strategy protocols for active learning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from probly.evaluation.active_learning.pool import ActiveLearningPool
    from probly.representation.array_like import ArrayLike


@runtime_checkable
class Estimator(Protocol):
    """Protocol for estimators usable by query strategies.

    Implement this for basic strategies (:class:`RandomQuery`,
    :class:`MarginSampling`). For :class:`UncertaintyQuery`, implement
    :class:`UncertaintyEstimator` instead. For :class:`BADGEQuery` with
    proper gradient embeddings, implement :class:`BadgeEstimator`.
    """

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


@runtime_checkable
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
