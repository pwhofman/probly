"""NumPy implementation of the active learning pool."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._common import from_dataset, query


@dataclass
class NumpyActiveLearningPool:
    """Active learning pool backed by NumPy arrays.

    Attributes:
        x_labeled: Feature matrix for the currently labeled training samples.
        y_labeled: Labels for the currently labeled training samples.
        x_unlabeled: Feature matrix for the unlabeled pool.
        y_unlabeled: Ground-truth labels for the unlabeled pool (revealed only on query).
        x_test: Feature matrix for the held-out test set.
        y_test: Labels for the held-out test set.
    """

    x_labeled: np.ndarray
    y_labeled: np.ndarray
    x_unlabeled: np.ndarray
    y_unlabeled: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    @property
    def n_labeled(self) -> int:
        """Number of currently labeled training samples."""
        return len(self.x_labeled)

    @property
    def n_unlabeled(self) -> int:
        """Number of samples remaining in the unlabeled pool."""
        return len(self.x_unlabeled)


@from_dataset.register(np.ndarray)
def _from_dataset_numpy(
    x: np.ndarray,
    y: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    initial_size: int,
    seed: int | None,
) -> NumpyActiveLearningPool:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x))
    labeled_idx = perm[:initial_size]
    unlabeled_idx = perm[initial_size:]
    return NumpyActiveLearningPool(
        x_labeled=x[labeled_idx],
        y_labeled=y[labeled_idx],
        x_unlabeled=x[unlabeled_idx],
        y_unlabeled=y[unlabeled_idx],
        x_test=x_test,
        y_test=y_test,
    )


@query.register(NumpyActiveLearningPool)
def _query_numpy(pool: NumpyActiveLearningPool, indices: np.ndarray) -> None:
    pool.x_labeled = np.concatenate([pool.x_labeled, pool.x_unlabeled[indices]], axis=0)
    pool.y_labeled = np.concatenate([pool.y_labeled, pool.y_unlabeled[indices]], axis=0)
    mask = np.ones(len(pool.x_unlabeled), dtype=bool)
    mask[indices] = False
    pool.x_unlabeled = pool.x_unlabeled[mask]
    pool.y_unlabeled = pool.y_unlabeled[mask]
