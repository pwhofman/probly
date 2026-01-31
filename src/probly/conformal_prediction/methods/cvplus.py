"""Cross-validation+ (CV+) implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from probly.conformal_prediction.methods.common import Predictor
    from probly.conformal_prediction.methods.jackknifeplus import (
        IntervalFunc,
        ScoreFunc,
    )

from probly.conformal_prediction.methods.jackknifeplus import (
    JackknifePlusClassifier,
    JackknifePlusRegressor,
)


class CVPlusRegressor(JackknifePlusRegressor):
    """CV+ Regressor (Defaults to 5-Fold)."""

    def __init__(
        self,
        model_factory: Callable[[], Predictor],
        cv: int = 5,
        random_state: int | None = None,
        score_func: ScoreFunc | None = None,
        interval_func: IntervalFunc | None = None,
    ) -> None:
        """Initialize CV+ Regressor."""
        super().__init__(model_factory, cv, random_state, score_func, interval_func)


class CVPlusClassifier(JackknifePlusClassifier):
    """CV+ Classifier (Defaults to 5-Fold)."""

    def __init__(
        self,
        model_factory: Callable[[], Predictor],
        cv: int = 5,
        random_state: int | None = None,
        use_accretive: bool = False,
        score_func: ScoreFunc | None = None,
    ) -> None:
        """Initialize CV+ Classifier."""
        super().__init__(model_factory, cv, random_state, use_accretive, score_func)
