"""sklearn implementation of the efficient credal prediction method."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

from probly.predictor import LogitClassifier, predict_raw

from ._common import efficient_credal_prediction_generator


@efficient_credal_prediction_generator.register(BaseEstimator)
class SklearnEfficientCredalPredictor(BaseEstimator):
    """sklearn estimator that wraps a fitted classifier and stores credal bounds."""

    def __init__(
        self,
        predictor: BaseEstimator,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
    ) -> None:
        """Initialize the predictor.

        Args:
            predictor: The fitted base classifier.
            lower: Per-class lower logit offsets of shape ``(num_classes,)``.
            upper: Per-class upper logit offsets of shape ``(num_classes,)``.
        """
        self.predictor = predictor
        self.lower = lower
        self.upper = upper


@predict_raw.register(SklearnEfficientCredalPredictor)
def sklearn_efficient_credal_predict_raw(
    predictor: SklearnEfficientCredalPredictor, /, *args: object, **kwargs: object
) -> np.ndarray:
    """Return the logits of the wrapped classifier with shape ``(N, num_classes)``."""
    # The LogitClassifier tag set by the transformation is a per-instance registration, which a
    # directly constructed or unpickled wrapper lacks.
    LogitClassifier.register_instance(predictor.predictor)
    logits = np.asarray(predict_raw(predictor.predictor, *args, **kwargs), dtype=float)
    if logits.ndim == 1:
        # A binary decision_function returns one margin per sample; expand it to two columns.
        logits = np.stack([np.zeros_like(logits), logits], axis=-1)
    return logits
