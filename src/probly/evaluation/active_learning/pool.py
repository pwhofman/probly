"""ActiveLearningPool: manages labeled, unlabeled, and test data for active learning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ActiveLearningPool:
    """Manages the labeled/unlabeled/test data split for a pool-based active learning run.

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

    @staticmethod
    def from_dataset(
        x: np.ndarray,
        y: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        initial_size: int,
        seed: int | None,
    ) -> ActiveLearningPool:
        """Create a pool by randomly splitting training data into labeled and unlabeled subsets.

        Uses ``np.random.default_rng(seed)`` followed by ``rng.permutation`` so that
        the split is fully reproducible for a given seed.

        Args:
            x: Full training feature matrix.
            y: Full training label array.
            x_test: Test feature matrix (kept as-is).
            y_test: Test label array (kept as-is).
            initial_size: Number of samples to place in the initial labeled set.
            seed: Seed for the random number generator.  Pass ``None`` for a
                non-deterministic split.

        Returns:
            A new ``ActiveLearningPool`` with ``initial_size`` labeled samples and
            the remainder in the unlabeled pool.
        """
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(x))
        labeled_idx = perm[:initial_size]
        unlabeled_idx = perm[initial_size:]
        return ActiveLearningPool(
            x_labeled=x[labeled_idx],
            y_labeled=y[labeled_idx],
            x_unlabeled=x[unlabeled_idx],
            y_unlabeled=y[unlabeled_idx],
            x_test=x_test,
            y_test=y_test,
        )

    def query(self, indices: np.ndarray) -> None:
        """Move samples from the unlabeled pool into the labeled set.

        The queried samples are appended to ``x_labeled`` and ``y_labeled`` and
        then removed from ``x_unlabeled`` and ``y_unlabeled`` via a boolean mask.

        Args:
            indices: Integer indices into the current ``x_unlabeled`` array
                identifying which samples to label.
        """
        self.x_labeled = np.concatenate([self.x_labeled, self.x_unlabeled[indices]], axis=0)
        self.y_labeled = np.concatenate([self.y_labeled, self.y_unlabeled[indices]], axis=0)
        mask = np.ones(len(self.x_unlabeled), dtype=bool)
        mask[indices] = False
        self.x_unlabeled = self.x_unlabeled[mask]
        self.y_unlabeled = self.y_unlabeled[mask]
