"""Active learning step iterator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from probly.evaluation.active_learning.pool import ActiveLearningPool
    from probly.evaluation.active_learning.strategies import Estimator, QueryStrategy


@dataclass
class ALState:
    """State yielded after each active learning iteration.

    Attributes:
        iteration: Current iteration number (0 = after initial training).
        pool: The active learning pool (mutated in place).
        estimator: The estimator trained on the current labeled set.
    """

    iteration: int
    pool: ActiveLearningPool
    estimator: Estimator


def active_learning_steps(
    pool: ActiveLearningPool,
    estimator: Estimator,
    query_strategy: QueryStrategy,
    query_size: int = 1000,
    n_iterations: int = 10,
) -> Iterator[ALState]:
    """Yield AL state after initial training and each query-retrain cycle.

    Args:
        pool: Data pool managing labeled/unlabeled splits.
        estimator: Model to train and query. Must implement fit/predict/predict_proba.
        query_strategy: Strategy for selecting which unlabeled samples to query.
        query_size: Number of samples to query per iteration.
        n_iterations: Maximum number of query-retrain iterations.

    Yields:
        ALState after initial training (iteration=0) and after each subsequent
        query-retrain cycle.
    """
    estimator.fit(pool.x_labeled, pool.y_labeled)
    yield ALState(iteration=0, pool=pool, estimator=estimator)

    for i in range(n_iterations):
        if pool.n_unlabeled == 0:
            break
        effective_n = min(query_size, pool.n_unlabeled)
        indices = query_strategy.select(estimator, pool, n=effective_n)
        pool.query(indices)
        estimator.fit(pool.x_labeled, pool.y_labeled)
        yield ALState(iteration=i + 1, pool=pool, estimator=estimator)
