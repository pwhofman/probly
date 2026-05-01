"""sklearn calibration wrappers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from packaging.version import Version
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from probly.calibrator import calibrate
from probly.predictor import LogitClassifier
from probly.transformation.calibration import CalibrationPredictor

from ._common import CalibrationMethodConfig, calibration_generator


def _extract_calibration_inputs(
    y_calib: object,
    calib_args: tuple[object, ...],
) -> tuple[object, object]:
    if not calib_args:
        msg = "Expected calibration inputs after y_calib, but none were provided."
        raise ValueError(msg)
    if len(calib_args) != 1:
        msg = f"Expected exactly one calibration input after y_calib, but got {len(calib_args)} positional arguments."
        raise ValueError(msg)
    return calib_args[0], y_calib


@calibrate.register(CalibratedClassifierCV)
def calibrate_sklearn_calibrated_classifier_cv(
    predictor: CalibratedClassifierCV,
    y_calib: object,
    *calib_args: object,
    **calib_kwargs: object,
) -> CalibratedClassifierCV:
    """Delegate generic calibrate(...) calls to sklearn's fit(X, y)."""
    x_calib, y_true = _extract_calibration_inputs(y_calib, calib_args)
    predictor.fit(x_calib, y_true, **calib_kwargs)
    return predictor


@dataclass(slots=True)
class _VectorScalingState:
    temperature: np.ndarray
    bias: np.ndarray


@LogitClassifier.register
class SklearnIdentityLogitEstimator(ClassifierMixin, BaseEstimator):
    """Pass-through sklearn estimator returning provided logits unchanged."""

    classes_: np.ndarray

    def __init__(self) -> None:
        """Initialize unfitted state."""
        super().__init__()
        self.is_fitted_ = True  # This estimator is always "fitted" since it has no parameters to fit.

    def fit(self, x: object, y: object) -> SklearnIdentityLogitEstimator:
        """Record fitted-state metadata for sklearn compatibility.

        Normally fit is not needed, since the primary use of this estimator is to pass-through logits from an
        already-fitted predictor via :meth:`decision_function`.

        The only use of this method is to populate the `classes_` attribute required by :meth:`predict`.
        """
        logits = np.asarray(x, dtype=float)
        if logits.ndim < 2:
            msg = f"Expected logits with class axis, got shape {logits.shape}."
            raise ValueError(msg)
        labels = np.asarray(y).reshape(-1)
        self.classes_ = np.asarray(np.unique(labels))
        return self

    def decision_function(self, x: object) -> np.ndarray:
        """Return input logits unchanged."""
        logits = np.asarray(x, dtype=float)

        if not hasattr(self, "classes_"):
            self.classes_ = np.arange(np.maximum(2, logits.shape[-1] if logits.ndim >= 2 else 1))

        return logits

    def predict_proba(self, x: object) -> np.ndarray:
        """Return probabilities corresponding to input logits."""
        logits = self.decision_function(x)
        if logits.ndim < 2:
            return np.stack([1.0 - expit(logits), expit(logits)], axis=-1)
        return np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))

    def predict(self, x: object) -> np.ndarray:
        """Predict labels by argmax over provided logits."""
        logits = self.decision_function(x)
        if logits.ndim < 2:
            return (logits > 0).astype(int)
        indices = np.argmax(logits, axis=-1)
        return self.classes_[indices]


class SklearnVectorScalingPredictor(BaseEstimator, CalibrationPredictor):
    """sklearn estimator wrapper implementing vector scaling calibration."""

    predictor: BaseEstimator
    classes_: np.ndarray
    _state: _VectorScalingState | None

    def __init__(self, predictor: BaseEstimator, num_classes: int | None = None, max_iter: int = 256) -> None:
        """Initialize vector scaling calibrator with unfitted state."""
        self.predictor = predictor
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.classes_ = np.empty((0,), dtype=np.int64)
        self._state = None

    @property
    def estimator(self) -> BaseEstimator:
        """Alias to sklearn's conventional attribute name for wrapped estimators."""
        return self.predictor

    @estimator.setter
    def estimator(self, value: BaseEstimator) -> None:
        """Set wrapped estimator via sklearn-conventional attribute alias."""
        self.predictor = value

    @property
    def is_calibrated_(self) -> bool:
        """Return whether vector scaling state was fitted."""
        return self._state is not None

    def _require_calibrated(self) -> _VectorScalingState:
        state = self._state
        if state is None:
            msg = "Calibration wrapper is not calibrated. Please call calibrate() or fit() before prediction."
            raise ValueError(msg)
        return state

    @property
    def temperature(self) -> np.ndarray | None:
        """Return calibrated temperature parameters if available."""
        state = self._state
        if state is None:
            return None
        return np.array(state.temperature, copy=True)

    @property
    def bias(self) -> np.ndarray | None:
        """Return calibrated bias parameters if available."""
        state = self._state
        if state is None:
            return None
        return np.array(state.bias, copy=True)

    def _extract_logits(self, x: object) -> np.ndarray:
        decision_function = getattr(self.predictor, "decision_function", None)
        if decision_function is not None and callable(decision_function):
            logits = np.asarray(decision_function(x), dtype=float)
            if logits.ndim == 1:
                logits = np.stack([np.zeros_like(logits), logits], axis=-1)
            return logits

        predict_proba = getattr(self.predictor, "predict_proba", None)
        if predict_proba is not None and callable(predict_proba):
            probs = np.asarray(predict_proba(x), dtype=float)
            probs = np.clip(probs, 1e-12, 1.0)
            if probs.ndim < 1:
                msg = f"Expected probability outputs with at least one dimension, got shape {probs.shape}."
                raise ValueError(msg)
            if probs.ndim == 1:
                probs = np.stack([probs, 1.0 - probs], axis=-1)
            if probs.shape[-1] == 1:
                p = probs[..., 0]
                probs = np.stack([1.0 - p, p], axis=-1)
            if probs.shape[-1] == 2:
                logit = np.log(probs[..., 1]) - np.log(probs[..., 0])
                return np.stack([np.zeros_like(logit), logit], axis=-1)
            return np.log(probs)

        msg = f"Wrapped estimator {type(self.predictor)} provides neither decision_function nor predict_proba."
        raise AttributeError(msg)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        numerator = np.exp(shifted)
        denominator = np.sum(numerator, axis=-1, keepdims=True)
        return numerator / denominator

    @staticmethod
    def _affine_logits(logits: np.ndarray, temperature: np.ndarray, bias: np.ndarray) -> np.ndarray:
        return logits / temperature + bias

    @staticmethod
    def _encode_labels(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
        class_to_idx = {label: idx for idx, label in enumerate(classes.tolist())}
        return np.asarray([class_to_idx[label] for label in y], dtype=np.int64)

    def fit(self, x: object, y: object, **_fit_kwargs: object) -> SklearnVectorScalingPredictor:
        """Fit vector-scaling calibration parameters on calibration data."""
        logits = self._extract_logits(x)
        if logits.ndim < 2:
            msg = "Vector scaling expects logits with an explicit class axis and more than one class."
            raise ValueError(msg)
        num_classes = int(logits.shape[-1])
        if num_classes <= 1:
            msg = "Vector scaling expects logits with class dimension > 1."
            raise ValueError(msg)
        if self.num_classes is not None and self.num_classes != num_classes:
            msg = f"Expected {self.num_classes} classes for vector scaling, got {num_classes}."
            raise ValueError(msg)

        flat_logits = logits.reshape(-1, num_classes)

        y_arr = np.asarray(y)
        flat_y = y_arr.reshape(-1)
        if flat_y.size != flat_logits.shape[0]:
            msg = (
                "Calibration labels must match logits batch shape. "
                f"Got {flat_y.size} labels for {flat_logits.shape[0]} logits."
            )
            raise ValueError(msg)
        classes = np.asarray(getattr(self.predictor, "classes_", np.unique(y_arr)))
        if len(classes) != num_classes:
            msg = f"Number of estimator classes does not match logits class dimension: {len(classes)} vs {num_classes}."
            raise ValueError(msg)
        encoded_y = self._encode_labels(flat_y, classes)

        def objective(params: np.ndarray) -> float:
            log_temperature = params[:num_classes]
            bias = params[num_classes:]
            temperature = np.exp(log_temperature)
            affine = self._affine_logits(flat_logits, temperature, bias)
            log_probs = affine - logsumexp(affine, axis=-1, keepdims=True)
            return float(-np.mean(log_probs[np.arange(encoded_y.size), encoded_y]))

        initial = np.zeros(2 * num_classes, dtype=float)
        optimization = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )

        params = optimization.x
        temperature = np.exp(params[:num_classes])
        bias = params[num_classes:]
        self.classes_ = classes
        self._state = _VectorScalingState(temperature=temperature, bias=bias)
        return self

    def calibrate(self, y_calib: object, *calib_args: object, **calib_kwargs: object) -> SklearnVectorScalingPredictor:
        """Calibrate vector-scaling parameters using probly's generic argument order."""
        x_calib, y_true = _extract_calibration_inputs(y_calib, calib_args)
        return self.fit(x_calib, y_true, **calib_kwargs)

    def predict_logits(self, x: object) -> np.ndarray:
        """Predict calibrated logits for input samples."""
        state = self._require_calibrated()
        logits = self._extract_logits(x)
        return self._affine_logits(logits, state.temperature, state.bias)

    def predict_proba(self, x: object) -> np.ndarray:
        """Predict calibrated probabilities for input samples."""
        return self._softmax(self.predict_logits(x))

    def predict(self, x: object) -> np.ndarray:
        """Predict labels based on calibrated probabilities."""
        probabilities = self.predict_proba(x)
        if self.classes_.size == 0:
            msg = "Calibration wrapper is missing classes_."
            raise ValueError(msg)
        indices = np.argmax(probabilities, axis=-1)
        return self.classes_[indices]


@calibration_generator.register(BaseEstimator)
def generate_sklearn_scaling_calibrator(
    base: BaseEstimator,
    config: CalibrationMethodConfig,
) -> BaseEstimator:
    """Create sklearn scaling calibrators from configuration."""
    if config.method in {"temperature", "platt", "isotonic"}:
        method = "sigmoid" if config.method == "platt" else config.method
        if method == "temperature" and Version(sklearn.__version__) < Version("1.8.0"):
            msg = "Temperature scaling calibration requires scikit-learn 1.8.0 or later."
            raise ValueError(msg)

        return CalibratedClassifierCV(estimator=FrozenEstimator(base), method=method)
    return SklearnVectorScalingPredictor(base, num_classes=config.num_classes)
