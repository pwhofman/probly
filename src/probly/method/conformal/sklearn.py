"""sklearn conformal predictor wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

    def fit(self, *args: object, **kwargs: object) -> SklearnConformalSetPredictor[In, Out]:
        """Fit the wrapped estimator and return self."""
        fit_method = getattr(self.predictor, "fit", None)
        if fit_method is None or not callable(fit_method):
            msg = f"Wrapped predictor {type(self.predictor)} has no fit method."
            raise AttributeError(msg)
        fit_method(*args, **kwargs)
        return self

    def predict(self, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        """Forward ``predict`` to the wrapped estimator."""
        predict_method = getattr(self.predictor, "predict", None)
        if predict_method is None or not callable(predict_method):
            msg = f"Wrapped predictor {type(self.predictor)} has no predict method."
            raise AttributeError(msg)
        return predict_method(*args, **kwargs)
