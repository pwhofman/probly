"""Query strategies for pool-based active learning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from probly.evaluation.active_learning.pool import ActiveLearningPool


@runtime_checkable
class Estimator(Protocol):
    """Protocol for estimators usable by query strategies."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the estimator on labeled data."""
        ...

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        ...

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
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


def _badge_select(embeddings: np.ndarray, probs: np.ndarray, n: int) -> np.ndarray:
    """Select n indices via BADGE k-means++ initialization.

    Computes gradient embeddings from predicted probabilities and embeddings,
    then runs k-means++ to select a diverse uncertain batch.

    Args:
        embeddings: Feature embeddings of shape (n_pool, emb_dim).
        probs: Predicted class probabilities of shape (n_pool, n_classes).
        n: Number of indices to select.

    Returns:
        Array of n integer indices.
    """
    predicted_class = np.argmax(probs, axis=1)
    p_predicted = probs[np.arange(len(probs)), predicted_class]
    grad_embeddings = embeddings * (1 - p_predicted)[:, np.newaxis]

    rng = np.random.default_rng()
    n_pool = len(grad_embeddings)

    # k-means++ initialization
    first = int(rng.integers(0, n_pool))
    chosen: list[int] = [first]

    for _ in range(1, n):
        dists = cdist(grad_embeddings, grad_embeddings[chosen], metric="sqeuclidean")
        min_dists = dists.min(axis=1)
        min_dists[chosen] = 0.0
        total = min_dists.sum()
        if total == 0.0:
            # All remaining distances are zero; pick uniformly from unchosen
            remaining = np.setdiff1d(np.arange(n_pool), chosen)
            if len(remaining) == 0:
                break
            next_idx = int(rng.choice(remaining))
        else:
            probs_sample = min_dists / total
            next_idx = int(rng.choice(n_pool, p=probs_sample))
        chosen.append(next_idx)

    return np.array(chosen)


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
        sorted_probs = np.sort(probs, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        return np.argpartition(margin, n)[:n]


class EntropyQuery:
    """Selects unlabeled samples with the highest Shannon entropy of predicted probs.

    Higher entropy means the model is more uncertain about the predicted class
    distribution over all classes.
    """

    def select(self, estimator: Estimator, pool: ActiveLearningPool, n: int) -> np.ndarray:
        """Return n indices with the highest predicted class entropy.

        Args:
            estimator: Fitted estimator with a predict_proba method.
            pool: The current active learning pool.
            n: Number of samples to select.

        Returns:
            Array of n integer indices into pool.x_unlabeled.
        """
        n = min(n, pool.n_unlabeled)
        probs = estimator.predict_proba(pool.x_unlabeled)
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs), axis=1)
        return np.argpartition(entropy, -n)[-n:]


class MutualInfoQuery:
    """Selects samples with the highest estimator-provided uncertainty scores.

    Delegates scoring to estimator.uncertainty_scores(), which may implement
    mutual information or any other UQ-based measure from probly.
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
        return np.argpartition(scores, -n)[-n:]


class UncertaintyQuery:
    """Selects samples with the highest estimator-provided uncertainty scores.

    Like MutualInfoQuery, delegates scoring to estimator.uncertainty_scores().
    Suitable for any probly UQ method that provides its own per-sample scoring.
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
        return np.argpartition(scores, -n)[-n:]


class BADGEQuery:
    """Selects a diverse uncertain batch via BADGE gradient embedding k-means++.

    Implements Batch Active learning by Diverse Gradient Embeddings (Ash et al.,
    2020). If the estimator has an embed() method it is used for embeddings;
    otherwise pool.x_unlabeled is used directly.
    """

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
        embeddings = cast("Any", estimator).embed(pool.x_unlabeled) if hasattr(estimator, "embed") else pool.x_unlabeled
        return _badge_select(embeddings, probs, n)
