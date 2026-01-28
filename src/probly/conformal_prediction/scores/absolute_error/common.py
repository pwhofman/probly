"""Absolute Error Score implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from lazy_dispatch.isinstance import LazyType
    from probly.conformal_prediction.methods.common import Predictor

from lazy_dispatch.singledispatch import lazydispatch
from probly.conformal_prediction.scores.common import RegressionScore


@lazydispatch
def absolute_error_score_func(y_true: Any, y_pred: Any) -> Any:  # noqa: ANN401
    """Compute absolute error |y - y_hat|."""
    return abs(y_true - y_pred)


def register(cls: LazyType, func: Callable) -> None:
    """Register an implementation for a specific type."""
    absolute_error_score_func.register(cls=cls, func=func)


class AbsoluteErrorScore(RegressionScore):
    """Standard absolute residual score: |y - y_hat|."""

    def __init__(self, model: Predictor) -> None:
        """Initialize with a prediction model."""
        super().__init__(model=model, score_func=absolute_error_score_func)
