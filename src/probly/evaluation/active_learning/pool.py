"""ActiveLearningPool: manages labeled, unlabeled, and test data for active learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
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

    x_labeled: torch.Tensor
    y_labeled: torch.Tensor
    x_unlabeled: torch.Tensor
    y_unlabeled: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor

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
        x: torch.Tensor,
        y: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        initial_size: int,
        seed: int | None,
    ) -> ActiveLearningPool:
        """Create a pool by randomly splitting training data into labeled and unlabeled subsets.

        Uses ``torch.randperm`` with a seeded generator so that the split is
        fully reproducible for a given seed.

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
        g = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
        perm = torch.randperm(len(x), generator=g)
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
        idx = torch.from_numpy(indices).long()
        self.x_labeled = torch.cat([self.x_labeled, self.x_unlabeled[idx]], dim=0)
        self.y_labeled = torch.cat([self.y_labeled, self.y_unlabeled[idx]], dim=0)
        mask = torch.ones(len(self.x_unlabeled), dtype=torch.bool)
        mask[idx] = False
        self.x_unlabeled = self.x_unlabeled[mask]
        self.y_unlabeled = self.y_unlabeled[mask]
