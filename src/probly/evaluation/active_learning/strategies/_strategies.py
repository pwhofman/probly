"""Concrete query strategy classes for active learning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._scores import entropy_score, least_confident_score, margin_score
from ._selection import random_select, topk_select

if TYPE_CHECKING:
    from probly.evaluation.active_learning.pool import ActiveLearningPool
    from probly.representation.array_like import ArrayLike

    from ._protocols import Estimator, UncertaintyEstimator


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


class EntropySampling:
    """Selects unlabeled samples with the highest prediction entropy.

    High entropy indicates the model's predicted distribution is spread across
    classes, making these samples informative to label.
    """

    def select(self, estimator: Estimator, pool: ActiveLearningPool, n: int) -> ArrayLike:
        """Return n indices with the highest prediction entropy."""
        n = min(n, pool.n_unlabeled)
        scores = entropy_score(estimator.predict_proba(pool.x_unlabeled))
        return topk_select(scores, n)


class LeastConfidentSampling:
    """Selects unlabeled samples where the model is least confident in its top prediction.

    A low maximum probability indicates the model is uncertain overall,
    making these samples informative to label.
    """

    def select(self, estimator: Estimator, pool: ActiveLearningPool, n: int) -> ArrayLike:
        """Return n indices with the lowest maximum class probability."""
        n = min(n, pool.n_unlabeled)
        scores = least_confident_score(estimator.predict_proba(pool.x_unlabeled))
        return topk_select(scores, n)


class MarginSampling:
    """Selects unlabeled samples with the smallest margin between top-2 class probs.

    A small margin indicates the model is uncertain between its top two predicted
    classes, making these samples most informative to label.
    """

    def select(self, estimator: Estimator, pool: ActiveLearningPool, n: int) -> ArrayLike:
        """Return n indices with the smallest margin in predicted probabilities."""
        n = min(n, pool.n_unlabeled)
        scores = margin_score(estimator.predict_proba(pool.x_unlabeled))
        return topk_select(scores, n)


class UncertaintyQuery:
    """Selects samples with the highest estimator-provided uncertainty scores.

    Delegates scoring to estimator.uncertainty_scores(). Suitable for any probly
    UQ method that provides its own per-sample scoring (e.g. mutual information
    or any other UQ-based measure).
    """

    def select(self, estimator: UncertaintyEstimator, pool: ActiveLearningPool, n: int) -> ArrayLike:
        """Return n indices with the highest uncertainty scores from the estimator."""
        n = min(n, pool.n_unlabeled)
        scores = estimator.uncertainty_scores(pool.x_unlabeled)
        return topk_select(scores, n)
