"""ActiveLearningPool protocol and dispatched factory/mutation functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flextype import flexdispatch

if TYPE_CHECKING:
    import numpy as np

    from probly.representation.array_like import ArrayLike


@runtime_checkable
class ActiveLearningPool(Protocol):
    """Protocol for backend-specific active learning pools.

    Attributes:
        x_labeled: Feature matrix for the currently labeled training samples.
        y_labeled: Labels for the currently labeled training samples.
        x_unlabeled: Feature matrix for the unlabeled pool.
        y_unlabeled: Ground-truth labels for the unlabeled pool (revealed only on query).
        x_test: Feature matrix for the held-out test set.
        y_test: Labels for the held-out test set.
    """

    @property
    def x_labeled(self) -> ArrayLike: ...
    @property
    def y_labeled(self) -> ArrayLike: ...
    @property
    def x_unlabeled(self) -> ArrayLike: ...
    @property
    def y_unlabeled(self) -> ArrayLike: ...
    @property
    def x_test(self) -> ArrayLike: ...
    @property
    def y_test(self) -> ArrayLike: ...

    @property
    def n_labeled(self) -> int: ...
    @property
    def n_unlabeled(self) -> int: ...


@flexdispatch
def from_dataset(
    x: object,
    y: object,
    x_test: object,
    y_test: object,
    initial_size: int,
    seed: int | None,
) -> ActiveLearningPool:
    """Create a pool by randomly splitting training data into labeled and unlabeled subsets.

    Dispatches on the type of ``x`` to select the appropriate backend.

    Args:
        x: Full training feature matrix.
        y: Full training label array.
        x_test: Test feature matrix (kept as-is).
        y_test: Test label array (kept as-is).
        initial_size: Number of samples to place in the initial labeled set.
        seed: Seed for the random number generator. Pass ``None`` for a
            non-deterministic split.

    Returns:
        A backend-specific ``ActiveLearningPool`` with ``initial_size`` labeled
        samples and the remainder in the unlabeled pool.
    """
    msg = f"No from_dataset implementation registered for type {type(x)}"
    raise NotImplementedError(msg)


@flexdispatch
def query(pool: object, indices: np.ndarray) -> None:
    """Move samples from the unlabeled pool into the labeled set.

    Dispatches on the type of ``pool``.

    Args:
        pool: The active learning pool to mutate in-place.
        indices: Integer indices into the current unlabeled array identifying
            which samples to label.
    """
    msg = f"No query implementation registered for type {type(pool)}"
    raise NotImplementedError(msg)
