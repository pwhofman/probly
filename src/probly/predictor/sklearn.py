"""sklearn-specific predictor dispatch helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.neighbors import NearestCentroid

from ._common import (
    BernoulliDistributionPredictor,
    BernoulliLogitDistributionPredictor,
    CategoricalDistributionPredictor,
    GaussianDistributionPredictor,
    LogitDistributionPredictor,
    predict_raw,
)

GaussianDistributionPredictor.register(GaussianProcessRegressor)
LogitDistributionPredictor.register(GaussianProcessClassifier)

_SAFE_DECISION_FUNCTION_TYPES = (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    LinearDiscriminantAnalysis,
    LogisticRegression,
    LogisticRegressionCV,
    NearestCentroid,
    QuadraticDiscriminantAnalysis,
)


def _callable_attribute(obj: object, name: str) -> Any | None:  # noqa: ANN401
    try:
        attr = getattr(obj, name)
    except AttributeError:
        return None
    return attr if callable(attr) else None


def _probabilities_to_logits(probabilities: object) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    return np.log(np.clip(probs, np.finfo(float).tiny, 1.0))


def _extract_binary_probability(probabilities: object) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim >= 1 and probs.shape[-1] <= 2:
        return probs[..., -1]
    return probs


def _has_safe_decision_function(predictor: BaseEstimator) -> bool:
    if isinstance(predictor, _SAFE_DECISION_FUNCTION_TYPES):
        return True
    return isinstance(predictor, SGDClassifier) and predictor.loss == "log_loss"


def _sklearn_logit_prediction[**In](predictor: BaseEstimator, *args: In.args, **kwargs: In.kwargs) -> np.ndarray:
    decision_function = _callable_attribute(predictor, "decision_function")
    if decision_function is not None and _has_safe_decision_function(predictor):
        return np.asarray(decision_function(*args, **kwargs), dtype=float)

    predict_log_proba = _callable_attribute(predictor, "predict_log_proba")
    if predict_log_proba is not None:
        return np.asarray(predict_log_proba(*args, **kwargs), dtype=float)

    predict_proba = _callable_attribute(predictor, "predict_proba")
    if predict_proba is not None:
        return _probabilities_to_logits(predict_proba(*args, **kwargs))

    msg = (
        f"Predictor of type {type(predictor)} is registered as a LogitDistributionPredictor, "
        "but sklearn cannot produce meaningful logits from a known-safe decision_function, predict_log_proba, "
        "or predict_proba. Register it as a probabilistic classifier instead, or use a calibrated/probabilistic "
        "estimator."
    )
    raise NotImplementedError(msg)


def _sklearn_binary_logit_prediction[**In](predictor: BaseEstimator, *args: In.args, **kwargs: In.kwargs) -> np.ndarray:
    logits = _sklearn_logit_prediction(predictor, *args, **kwargs)
    if logits.ndim >= 1 and logits.shape[-1] <= 2:
        return logits[..., 1] - logits[..., -1]
    return logits


@predict_raw.register(BaseEstimator)
def predict_sklearn[**In](predictor: BaseEstimator, /, *args: In.args, **kwargs: In.kwargs) -> Any:  # noqa: ANN401
    """Predict for sklearn estimators."""
    if isinstance(predictor, BernoulliLogitDistributionPredictor):
        return _sklearn_binary_logit_prediction(predictor, *args, **kwargs)

    if isinstance(predictor, LogitDistributionPredictor):
        return _sklearn_logit_prediction(predictor, *args, **kwargs)

    if isinstance(predictor, BernoulliDistributionPredictor):
        predict_proba = _callable_attribute(predictor, "predict_proba")
        if predict_proba is not None:
            return _extract_binary_probability(predict_proba(*args, **kwargs))

    if isinstance(predictor, CategoricalDistributionPredictor):
        predict_proba = _callable_attribute(predictor, "predict_proba")
        if predict_proba is not None:
            return predict_proba(*args, **kwargs)

    if isinstance(predictor, GaussianDistributionPredictor) and hasattr(predictor, "predict"):
        mean, std = predictor.predict(*args, return_std=True, **kwargs)
        return mean, np.square(std)

    predict = _callable_attribute(predictor, "predict")
    if predict is not None:
        return predict(*args, **kwargs)

    msg = f"No predict function registered for type {type(predictor)}"
    raise NotImplementedError(msg)
