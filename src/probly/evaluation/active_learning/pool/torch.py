"""PyTorch implementation of the active learning pool."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._common import _validate_initial_size, from_dataset, query


@dataclass
class TorchActiveLearningPool:
    """Active learning pool backed by PyTorch tensors.

    Attributes:
        x_labeled: Feature tensor for the currently labeled training samples.
        y_labeled: Label tensor for the currently labeled training samples.
        x_unlabeled: Feature tensor for the unlabeled pool.
        y_unlabeled: Ground-truth label tensor for the unlabeled pool (revealed only on query).
        x_test: Feature tensor for the held-out test set.
        y_test: Label tensor for the held-out test set.
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


@from_dataset.register(torch.Tensor)
def _from_dataset_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    initial_size: int,
    seed: int | None,
) -> TorchActiveLearningPool:
    _validate_initial_size(len(x), initial_size)
    g = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
    perm = torch.randperm(len(x), generator=g).to(x.device)
    labeled_idx = perm[:initial_size]
    unlabeled_idx = perm[initial_size:]
    return TorchActiveLearningPool(
        x_labeled=x[labeled_idx],
        y_labeled=y[labeled_idx],
        x_unlabeled=x[unlabeled_idx],
        y_unlabeled=y[unlabeled_idx],
        x_test=x_test,
        y_test=y_test,
    )


@query.register(TorchActiveLearningPool)
def _query_torch(pool: TorchActiveLearningPool, indices: torch.Tensor) -> None:
    idx = indices.long()
    pool.x_labeled = torch.cat([pool.x_labeled, pool.x_unlabeled[idx]], dim=0)
    pool.y_labeled = torch.cat([pool.y_labeled, pool.y_unlabeled[idx]], dim=0)
    mask = torch.ones(len(pool.x_unlabeled), dtype=torch.bool, device=pool.x_unlabeled.device)
    mask[idx] = False
    pool.x_unlabeled = pool.x_unlabeled[mask]
    pool.y_unlabeled = pool.y_unlabeled[mask]
