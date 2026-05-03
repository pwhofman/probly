"""Active learning module with composable pool, strategies, and iterator.

Typical workflow::

    from probly.evaluation.active_learning import (
        from_dataset, MarginSampling, active_learning_steps, compute_accuracy,
    )

    pool = from_dataset(x_train, y_train, x_test, y_test, initial_size=100, seed=42)
    for state in active_learning_steps(pool, estimator, MarginSampling(), query_size=50):
        acc = compute_accuracy(state.estimator.predict(state.pool.x_test), state.pool.y_test)

Pass numpy arrays for a numpy-backed pipeline or torch tensors for a
torch-backed pipeline. Each component dispatches independently on the
array type it receives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from probly.evaluation.active_learning.loop import (
    ALState as ALState,
    active_learning_steps as active_learning_steps,
)
from probly.evaluation.active_learning.metrics import (
    compute_accuracy as compute_accuracy,
    compute_ece as compute_ece,
    compute_nauc as compute_nauc,
)
from probly.evaluation.active_learning.pool import (
    ActiveLearningPool as ActiveLearningPool,
    from_dataset as from_dataset,
)
from probly.evaluation.active_learning.pool.array import (
    NumpyActiveLearningPool as NumpyActiveLearningPool,
)
from probly.evaluation.active_learning.strategies import (
    BadgeEstimator as BadgeEstimator,
    BADGEQuery as BADGEQuery,
    EntropySampling as EntropySampling,
    Estimator as Estimator,
    LeastConfidentSampling as LeastConfidentSampling,
    MarginSampling as MarginSampling,
    QueryStrategy as QueryStrategy,
    RandomQuery as RandomQuery,
    UncertaintyEstimator as UncertaintyEstimator,
    UncertaintyQuery as UncertaintyQuery,
)

if TYPE_CHECKING:
    from probly.evaluation.active_learning.pool.torch import (
        TorchActiveLearningPool as TorchActiveLearningPool,
    )

__all__ = [
    "ALState",
    "ActiveLearningPool",
    "BADGEQuery",
    "BadgeEstimator",
    "EntropySampling",
    "Estimator",
    "LeastConfidentSampling",
    "MarginSampling",
    "NumpyActiveLearningPool",
    "QueryStrategy",
    "RandomQuery",
    "TorchActiveLearningPool",
    "UncertaintyEstimator",
    "UncertaintyQuery",
    "active_learning_steps",
    "compute_accuracy",
    "compute_ece",
    "compute_nauc",
    "from_dataset",
]


def __getattr__(name: str) -> object:
    if name == "TorchActiveLearningPool":
        from probly.evaluation.active_learning.pool.torch import TorchActiveLearningPool  # noqa: PLC0415

        return TorchActiveLearningPool
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
