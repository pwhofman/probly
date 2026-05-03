"""sklearn conformal predictor wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from sklearn.base import BaseEstimator

from ._common import (
    _ConformalPredictorBase,
    conformal_generator,
)

if TYPE_CHECKING:
    import numpy as np

    from probly.conformal_scores._common import NonConformityScore


@conformal_generator.register(BaseEstimator)
class SklearnConformalSetPredictor[**In, Out](_ConformalPredictorBase[In, Out], BaseEstimator):
    """Base sklearn conformal wrapper forwarding sklearn APIs."""

    predictor: BaseEstimator

    def __init__(self, predictor: BaseEstimator, non_conformity_score: NonConformityScore[Out, np.ndarray]) -> None:
        """Initialize the sklearn conformal wrapper."""
        super().__init__(predictor, non_conformity_score)

    @property
    def estimator(self) -> BaseEstimator:
        """Alias to sklearn's conventional attribute name for wrapped estimators."""
        return self.predictor

    @estimator.setter
    def estimator(self, value: BaseEstimator) -> None:
        """Set wrapped estimator via sklearn-conventional attribute alias."""
        self.predictor = value

    def fit(
        self,
        x_calib: object,
        y_calib: object,
        *,
        alpha: float | None = None,
        **calib_kwargs: object,
    ) -> SklearnConformalSetPredictor[In, Out]:
        """Calibrate conformal state."""
        if alpha is None:
            msg = "alpha must be provided as a keyword argument when calling fit on sklearn conformal wrappers."
            raise ValueError(msg)
        return cast("Any", self).calibrate(alpha, y_calib, x_calib, **calib_kwargs)

    def predict(self, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        """Forward ``predict`` to the wrapped estimator."""
        predict_method = getattr(self.predictor, "predict", None)
        if predict_method is None or not callable(predict_method):
            msg = f"Wrapped predictor {type(self.predictor)} has no predict method."
            raise AttributeError(msg)
        return predict_method(*args, **kwargs)
