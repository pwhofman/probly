"""Active learning module with composable pool, strategies, and iterator."""

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
    Estimator as Estimator,
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


def __getattr__(name: str) -> object:
    if name == "TorchActiveLearningPool":
        from probly.evaluation.active_learning.pool.torch import TorchActiveLearningPool  # noqa: PLC0415

        return TorchActiveLearningPool
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
