"""Flax nnx logit calibration wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Self, override

from flax import nnx
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize as jax_minimize
import numpy as np
from scipy.optimize import minimize as scipy_minimize

from probly.predictor import BinaryLogitClassifier, LogitClassifier, predict_raw

from ._common import CalibrationMethodConfig, _CalibrationPredictorBase, calibration_generator

_ISOTONIC_MAX_KNOTS = 4096
_LBFGS_MAX_ITER = 256


def _reshape_binary_preds(preds: jax.Array) -> jax.Array:
    if preds.ndim >= 2 and preds.shape[-1] == 1:
        return preds.squeeze(-1)
    return preds


def _reshape_binary_labels(labels: jax.Array, expected_elements: int) -> jax.Array:
    flat_labels = labels.reshape(-1)
    if flat_labels.size != expected_elements:
        msg = (
            "Binary calibration labels must match logits batch size. "
            f"Got {flat_labels.size} labels for {expected_elements} logits."
        )
        raise ValueError(msg)
    return flat_labels


def _reshape_multiclass_labels(labels: jax.Array, batch_shape: tuple[int, ...]) -> jax.Array:
    expected_elements = math.prod(batch_shape) if batch_shape else 1
    flat_labels = labels.reshape(-1)
    if flat_labels.size != expected_elements:
        msg = (
            "Multiclass calibration labels must match logits batch size. "
            f"Got {flat_labels.size} labels for {expected_elements} logits."
        )
        raise ValueError(msg)
    return flat_labels


def _is_multiclass(logits: jax.Array, labels: jax.Array) -> bool:
    maybe_multiclass = logits.ndim >= 2 and logits.shape[-1] > 1
    batch_elements = math.prod(tuple(logits.shape[:-1])) if logits.ndim > 1 else 1
    labels_match_batch = labels.reshape(-1).size == batch_elements
    single_sample_multiclass = logits.ndim == 1 and logits.size > 1 and labels.reshape(-1).size == 1
    return (maybe_multiclass and labels_match_batch) or single_sample_multiclass


def _calibration_nll(scaled_logits: jax.Array, labels: jax.Array) -> jax.Array:
    if _is_multiclass(scaled_logits, labels):
        num_classes = int(scaled_logits.shape[-1])
        logits_2d = scaled_logits.reshape(-1, num_classes)
        labels_flat = _reshape_multiclass_labels(labels, tuple(scaled_logits.shape[:-1])).astype(jnp.int32)
        log_probs = jax.nn.log_softmax(logits_2d, axis=-1)
        picked = jnp.take_along_axis(log_probs, labels_flat[:, None], axis=-1).squeeze(-1)
        return -jnp.mean(picked)

    binary_logits = _reshape_binary_preds(scaled_logits).reshape(-1)
    labels_flat = _reshape_binary_labels(labels, binary_logits.size).astype(binary_logits.dtype)
    log_p1 = jax.nn.log_sigmoid(binary_logits)
    log_p0 = jax.nn.log_sigmoid(-binary_logits)
    return -jnp.mean(labels_flat * log_p1 + (1.0 - labels_flat) * log_p0)


def _prepare_binary_isotonic_inputs(preds: jax.Array, labels: jax.Array) -> tuple[jax.Array, jax.Array, bool]:
    if preds.ndim < 1:
        msg = f"Expected logits with at least one dimension, got shape {tuple(preds.shape)}."
        raise ValueError(msg)

    has_singleton_class_axis = preds.ndim >= 2 and preds.shape[-1] == 1
    if preds.ndim >= 2 and preds.shape[-1] > 1:
        msg = "Isotonic regression currently supports binary logits only (shape (...,) or (..., 1))."
        raise ValueError(msg)

    binary_logits = _reshape_binary_preds(preds)
    flat_logits = binary_logits.reshape(-1)
    flat_labels = labels.reshape(-1)
    if flat_labels.size != flat_logits.size:
        msg = (
            "Binary isotonic calibration labels must match logits batch size. "
            f"Got {flat_labels.size} labels for {flat_logits.size} logits."
        )
        raise ValueError(msg)
    return flat_logits, flat_labels, has_singleton_class_axis


def _apply_affine(logits: jax.Array, temperature: jax.Array, bias: jax.Array | None) -> jax.Array:
    if logits.ndim < 1:
        msg = f"Expected logits with at least one dimension, got shape {tuple(logits.shape)}."
        raise ValueError(msg)

    if temperature.size == 1:
        scaled = logits / temperature.reshape(())
    else:
        if logits.ndim < 2 or temperature.shape[-1] != logits.shape[-1]:
            msg = (
                "Temperature parameters do not match logits class dimension: "
                f"{temperature.shape[-1]} vs {logits.shape[-1] if logits.ndim >= 1 else 'unknown'}."
            )
            raise ValueError(msg)
        scaled = logits / temperature

    if bias is None:
        return scaled
    if bias.size == 1:
        return scaled + bias.reshape(())
    if logits.ndim < 2 or bias.shape[-1] != logits.shape[-1]:
        bias_dim = bias.shape[-1] if bias.ndim >= 1 else "unknown"
        logits_dim = logits.shape[-1] if logits.ndim >= 1 else "unknown"
        msg = f"Bias parameters do not match logits class dimension: {bias_dim} vs {logits_dim}."
        raise ValueError(msg)
    return scaled + bias


class FlaxIdentityLogitModel(nnx.Module):
    """Pass-through flax model returning provided logits unchanged."""

    def __call__(self, logits: jax.Array) -> jax.Array:
        """Return input logits unchanged."""
        return logits


LogitClassifier.register(FlaxIdentityLogitModel)
BinaryLogitClassifier.register(FlaxIdentityLogitModel)


class _FlaxCalibrationPredictorBase[**In](_CalibrationPredictorBase[In, jax.Array], nnx.Module, ABC):
    predictor: nnx.Module

    @abstractmethod
    def calibrate(self, y_calib: jax.Array, *calib_args: In.args, **calib_kwargs: In.kwargs) -> Self:
        """Calibrate model parameters on a held-out calibration split."""
        raise NotImplementedError

    @property
    def is_calibrated(self) -> bool:
        """Return whether calibration parameters were fitted."""
        value = getattr(self, "_is_calibrated", None)
        return value is not None and bool(value)

    def fit(self, x_calib: jax.Array, y_calib: jax.Array) -> Self:
        """Fit alias mapping sklearn-style argument order to calibrate."""
        return self.calibrate(y_calib, x_calib)  # ty:ignore[invalid-argument-type]


class FlaxAffineLogitCalibrationPredictor[**In](_FlaxCalibrationPredictorBase[In]):
    """Flax wrapper for parametric affine logit calibration methods."""

    def __init__(self, predictor: nnx.Module, config: CalibrationMethodConfig) -> None:
        """Initialize affine calibration state for the selected method."""
        super().__init__(predictor, config)
        self._temperature = self._initial_temperature_state(config)
        self._bias = self._initial_bias_state(config)
        self._is_calibrated = jnp.asarray(False)

    @staticmethod
    def _initial_temperature_state(config: CalibrationMethodConfig) -> jax.Array:
        if config.vector_scale and config.num_classes is not None:
            return jnp.full((config.num_classes,), jnp.nan, dtype=jnp.float32)
        return jnp.asarray(jnp.nan, dtype=jnp.float32)

    @staticmethod
    def _initial_bias_state(config: CalibrationMethodConfig) -> jax.Array:
        if not config.use_bias:
            return jnp.asarray(jnp.nan, dtype=jnp.float32)
        if config.vector_scale and config.num_classes is not None:
            return jnp.full((config.num_classes,), jnp.nan, dtype=jnp.float32)
        return jnp.asarray(jnp.nan, dtype=jnp.float32)

    @property
    def temperature(self) -> jax.Array | None:
        """Return fitted temperature parameters when available."""
        if not self.is_calibrated:
            return None
        return jnp.asarray(self._temperature)

    @property
    def bias(self) -> jax.Array | None:
        """Return fitted bias parameters when available."""
        if not self.config.use_bias or not self.is_calibrated:
            return None
        return jnp.asarray(self._bias)

    def _require_calibrated(self) -> tuple[jax.Array, jax.Array | None]:
        if not self.is_calibrated:
            msg = "Calibration wrapper is not calibrated. Please call calibrate() before prediction."
            raise ValueError(msg)

        temperature = jnp.asarray(self._temperature)
        if not self.config.use_bias:
            return temperature, None
        return temperature, jnp.asarray(self._bias)

    def _validate_vector_logits(self, logits: jax.Array, labels: jax.Array) -> int:
        if logits.ndim < 2 or logits.shape[-1] <= 1:
            msg = "Vector scaling expects logits with an explicit class axis and more than one class."
            raise ValueError(msg)
        num_classes = int(logits.shape[-1])
        expected_elements = math.prod(tuple(logits.shape[:-1])) if logits.ndim > 1 else 1
        if labels.reshape(-1).size != expected_elements:
            msg = (
                "Vector scaling labels must match logits batch size. "
                f"Got {labels.reshape(-1).size} labels for {expected_elements} logits."
            )
            raise ValueError(msg)
        if self.config.num_classes is not None and self.config.num_classes != num_classes:
            msg = f"Expected logits with {self.config.num_classes} classes, but got logits with {num_classes} classes."
            raise ValueError(msg)
        return num_classes

    @override
    def calibrate(self, y_calib: jax.Array, *calib_args: In.args, **calib_kwargs: In.kwargs) -> Self:
        """Calibrate affine scaling parameters on calibration data."""
        raw_logits = predict_raw(self.predictor, *calib_args, **calib_kwargs)
        if not isinstance(raw_logits, jax.Array):
            msg = f"Flax calibration expects jax logits, got {type(raw_logits)}"
            raise TypeError(msg)

        logits = jax.lax.stop_gradient(raw_logits)
        labels = y_calib if isinstance(y_calib, jax.Array) else jnp.asarray(y_calib)

        if logits.ndim < 1:
            msg = f"Expected logits with at least one dimension, got shape {tuple(logits.shape)}."
            raise ValueError(msg)

        num_classes = self._validate_vector_logits(logits, labels) if self.config.vector_scale else 1

        def objective(params: jax.Array) -> jax.Array:
            log_temperature = params[:num_classes]
            temperature = jnp.exp(log_temperature)
            bias = params[num_classes:] if self.config.use_bias else None
            scaled_logits = _apply_affine(logits, temperature, bias)
            return _calibration_nll(scaled_logits, labels)

        num_params = 2 * num_classes if self.config.use_bias else num_classes
        initial_params = jnp.zeros((num_params,), dtype=jnp.float32)

        result = jax_minimize(objective, initial_params, method="BFGS")
        fitted_params = result.x

        self._temperature = jnp.exp(fitted_params[:num_classes])
        if self.config.use_bias:
            self._bias = fitted_params[num_classes:]
        self._is_calibrated = jnp.asarray(True)

        return self

    def __call__(self, *args: In.args, **kwargs: In.kwargs) -> jax.Array:
        """Apply calibrated affine scaling to model logits."""
        raw_logits = predict_raw(self.predictor, *args, **kwargs)
        if not isinstance(raw_logits, jax.Array):
            msg = f"Flax calibration expects jax logits, got {type(raw_logits)}"
            raise TypeError(msg)
        temperature, bias = self._require_calibrated()
        return _apply_affine(raw_logits, temperature, bias)


class FlaxDirichletCalibrationPredictor[**In](_FlaxCalibrationPredictorBase[In]):
    """Flax wrapper applying a fitted Dirichlet calibration map to logits."""

    def __init__(self, predictor: nnx.Module, config: CalibrationMethodConfig) -> None:
        """Initialize NaN-filled Dirichlet calibration state for the given class count."""
        super().__init__(predictor, config)
        if config.num_classes is None or config.num_classes <= 1:
            msg = f"Dirichlet calibration expects num_classes > 1, but got {config.num_classes}."
            raise ValueError(msg)
        self.num_classes = int(config.num_classes)
        self.reg_lambda = float(config.reg_lambda)
        self.reg_mu = float(config.reg_lambda if config.reg_mu is None else config.reg_mu)
        self._dirichlet_weight = jnp.full((self.num_classes, self.num_classes), jnp.nan, dtype=jnp.float32)
        self._dirichlet_bias = jnp.full((self.num_classes,), jnp.nan, dtype=jnp.float32)
        self._is_calibrated = jnp.asarray(False)

    @property
    def weight(self) -> jax.Array | None:
        """Return the fitted weight matrix ``W`` when available."""
        if not self.is_calibrated:
            return None
        return jnp.asarray(self._dirichlet_weight)

    @property
    def bias(self) -> jax.Array | None:
        """Return the fitted bias ``b`` when available."""
        if not self.is_calibrated:
            return None
        return jnp.asarray(self._dirichlet_bias)

    def _require_calibrated(self) -> tuple[jax.Array, jax.Array]:
        if not self.is_calibrated:
            msg = "Calibration wrapper is not calibrated. Please call calibrate() before prediction."
            raise ValueError(msg)
        weight = jnp.asarray(self._dirichlet_weight)
        bias = jnp.asarray(self._dirichlet_bias)
        if weight.shape != (self.num_classes, self.num_classes) or bias.shape != (self.num_classes,):
            msg = (
                f"Dirichlet calibration state shapes {tuple(weight.shape)} and {tuple(bias.shape)} do not match "
                f"num_classes={self.num_classes}; the state was likely restored from a different configuration."
            )
            raise ValueError(msg)
        return weight, bias

    def _log_probabilities(self, *args: In.args, **kwargs: In.kwargs) -> jax.Array:
        raw_logits = predict_raw(self.predictor, *args, **kwargs)
        if not isinstance(raw_logits, jax.Array):
            msg = f"Flax Dirichlet calibration expects jax logits, got {type(raw_logits)}"
            raise TypeError(msg)
        if raw_logits.ndim < 2 or raw_logits.shape[-1] != self.num_classes:
            msg = (
                "Dirichlet calibration expects logits with an explicit class axis of size "
                f"{self.num_classes}, got shape {tuple(raw_logits.shape)}."
            )
            raise ValueError(msg)
        return jax.nn.log_softmax(raw_logits, axis=-1)

    @override
    def calibrate(self, y_calib: jax.Array, *calib_args: In.args, **calib_kwargs: In.kwargs) -> Self:
        """Calibrate the Dirichlet map on calibration data with ODIR regularization."""
        log_p = jax.lax.stop_gradient(self._log_probabilities(*calib_args, **calib_kwargs))
        labels = y_calib if isinstance(y_calib, jax.Array) else jnp.asarray(y_calib)
        labels_int = _reshape_multiclass_labels(labels, tuple(log_p.shape[:-1])).astype(jnp.int32)

        flat_log_p = log_p.reshape(-1, self.num_classes)
        if flat_log_p.shape[0] == 0:
            msg = "Dirichlet calibration requires a non-empty calibration set."
            raise ValueError(msg)
        if not bool(jnp.isfinite(flat_log_p).all()):
            msg = "Dirichlet calibration received non-finite log-probabilities from the predictor."
            raise ValueError(msg)
        if bool(jnp.any((labels_int < 0) | (labels_int >= self.num_classes))):
            msg = f"Dirichlet calibration labels must lie in [0, {self.num_classes})."
            raise ValueError(msg)

        num_weights = self.num_classes * self.num_classes
        off_diag_mask = ~jnp.eye(self.num_classes, dtype=bool)
        off_diag_count = self.num_classes * (self.num_classes - 1)

        def objective(params: jax.Array) -> jax.Array:
            weight = params[:num_weights].reshape(self.num_classes, self.num_classes)
            bias = params[num_weights:]
            calibrated = flat_log_p @ weight.T + bias
            loss = _calibration_nll(calibrated, labels_int)
            off_diagonal = jnp.where(off_diag_mask, weight, 0.0)
            loss = loss + self.reg_lambda * jnp.sum(off_diagonal**2) / off_diag_count
            return loss + self.reg_mu * jnp.sum(bias**2) / self.num_classes

        # jax.scipy.optimize.minimize's BFGS line search stalls on this objective and its
        # dense inverse-Hessian needs O(num_classes**4) memory, so optimize with scipy's
        # L-BFGS-B driven by jax gradients instead.
        value_and_grad = jax.jit(jax.value_and_grad(objective))

        def scipy_objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            value, grad = value_and_grad(jnp.asarray(params, dtype=flat_log_p.dtype))
            return float(value), np.asarray(grad, dtype=np.float64)

        initial_params = np.concatenate([np.eye(self.num_classes).reshape(-1), np.zeros(self.num_classes)])
        result = scipy_minimize(
            scipy_objective,
            initial_params,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": _LBFGS_MAX_ITER},
        )
        fitted_params = jnp.asarray(result.x, dtype=jnp.float32)

        self._dirichlet_weight = fitted_params[:num_weights].reshape(self.num_classes, self.num_classes)
        self._dirichlet_bias = fitted_params[num_weights:]
        self._is_calibrated = jnp.asarray(True)
        return self

    def __call__(self, *args: In.args, **kwargs: In.kwargs) -> jax.Array:
        """Apply the fitted Dirichlet calibration map and return calibrated logits."""
        weight, bias = self._require_calibrated()
        log_p = self._log_probabilities(*args, **kwargs)
        return log_p @ weight.T.astype(log_p.dtype) + bias.astype(log_p.dtype)


class FlaxIsotonicCalibrationPredictor[**In](_FlaxCalibrationPredictorBase[In]):
    """Flax wrapper for non-parametric isotonic calibration on binary logits."""

    def __init__(self, predictor: nnx.Module, config: CalibrationMethodConfig) -> None:
        """Initialize fixed-layout state for isotonic knot serialization."""
        super().__init__(predictor, config)
        self._isotonic_x_knots = jnp.full((_ISOTONIC_MAX_KNOTS,), jnp.nan, dtype=jnp.float32)
        self._isotonic_y_knots = jnp.full((_ISOTONIC_MAX_KNOTS,), jnp.nan, dtype=jnp.float32)
        self._isotonic_num_knots = jnp.asarray(0, dtype=jnp.int32)
        self._is_calibrated = jnp.asarray(False)

    @staticmethod
    def _fit_isotonic_knots(preds: jax.Array, labels: jax.Array) -> tuple[jax.Array, jax.Array]:
        try:
            from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415
        except ImportError as err:  # pragma: no cover - guarded by optional dependency
            msg = "Flax isotonic calibration requires scikit-learn to fit isotonic regression knots."
            raise ImportError(msg) from err

        x_np = np.asarray(preds, dtype=float)
        y_np = np.asarray(labels, dtype=float)
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(x_np, y_np)

        x_knots = jnp.asarray(model.X_thresholds_, dtype=jnp.float32)
        y_knots = jnp.asarray(model.y_thresholds_, dtype=jnp.float32)
        return x_knots, y_knots

    def _store_isotonic_knots(self, x_knots: jax.Array, y_knots: jax.Array) -> None:
        num_knots = int(x_knots.shape[0])
        if num_knots > _ISOTONIC_MAX_KNOTS:
            msg = (
                "Isotonic calibration produced more knots than supported by the fixed nnx state layout: "
                f"{num_knots} > {_ISOTONIC_MAX_KNOTS}."
            )
            raise ValueError(msg)

        x_buffer = jnp.full((_ISOTONIC_MAX_KNOTS,), jnp.nan, dtype=jnp.float32).at[:num_knots].set(x_knots)
        y_buffer = jnp.full((_ISOTONIC_MAX_KNOTS,), jnp.nan, dtype=jnp.float32).at[:num_knots].set(y_knots)

        self._isotonic_x_knots = x_buffer
        self._isotonic_y_knots = y_buffer
        self._isotonic_num_knots = jnp.asarray(num_knots, dtype=jnp.int32)
        self._is_calibrated = jnp.asarray(True)

    def _require_isotonic_knots(self) -> tuple[jax.Array, jax.Array]:
        if not self.is_calibrated:
            msg = "Calibration wrapper is not calibrated. Please call calibrate() before prediction."
            raise ValueError(msg)

        num_knots = int(self._isotonic_num_knots)
        if num_knots <= 0:
            msg = "Isotonic calibration has no fitted knots."
            raise ValueError(msg)
        return self._isotonic_x_knots[:num_knots], self._isotonic_y_knots[:num_knots]

    def _apply_isotonic(self, preds: jax.Array) -> jax.Array:
        flat_preds, _, had_singleton_axis = _prepare_binary_isotonic_inputs(preds, jnp.zeros_like(preds))
        x_knots, y_knots = self._require_isotonic_knots()

        x_knots = x_knots.astype(flat_preds.dtype)
        y_knots = y_knots.astype(flat_preds.dtype)

        if x_knots.shape[0] == 1:
            probs = jnp.broadcast_to(y_knots[0], flat_preds.shape)
        else:
            interior = x_knots[1:-1]
            interval_idx = jnp.searchsorted(interior, flat_preds, side="right")
            left_x = x_knots[interval_idx]
            right_x = x_knots[interval_idx + 1]
            left_y = y_knots[interval_idx]
            right_y = y_knots[interval_idx + 1]
            denominator = jnp.clip(right_x - left_x, min=1e-12)
            weight = jnp.clip((flat_preds - left_x) / denominator, min=0.0, max=1.0)
            probs = left_y + weight * (right_y - left_y)

        calibrated_probs = probs.reshape(_reshape_binary_preds(preds).shape)
        if had_singleton_axis:
            return calibrated_probs[..., None]
        return calibrated_probs

    @override
    def calibrate(self, y_calib: jax.Array, *calib_args: In.args, **calib_kwargs: In.kwargs) -> Self:
        """Calibrate isotonic parameters on calibration data."""
        raw_preds = predict_raw(self.predictor, *calib_args, **calib_kwargs)
        if not isinstance(raw_preds, jax.Array):
            msg = f"Flax calibration expects jax predictions, got {type(raw_preds)}"
            raise TypeError(msg)

        preds = jax.lax.stop_gradient(raw_preds)
        labels = y_calib if isinstance(y_calib, jax.Array) else jnp.asarray(y_calib)

        flat_preds, flat_labels, _ = _prepare_binary_isotonic_inputs(preds, labels)
        x_knots, y_knots = self._fit_isotonic_knots(flat_preds, flat_labels)
        self._store_isotonic_knots(x_knots, y_knots)
        return self

    def __call__(self, *args: In.args, **kwargs: In.kwargs) -> jax.Array:
        """Apply fitted isotonic calibration to model logits."""
        raw_preds = predict_raw(self.predictor, *args, **kwargs)
        if not isinstance(raw_preds, jax.Array):
            msg = f"Flax calibration expects jax logits, got {type(raw_preds)}"
            raise TypeError(msg)
        return self._apply_isotonic(raw_preds)


@calibration_generator.register(nnx.Module)
def generate_flax_scaling_calibrator(
    base: nnx.Module, config: CalibrationMethodConfig
) -> _FlaxCalibrationPredictorBase:
    """Create flax calibration wrappers from method configuration."""
    if config.method == "isotonic":
        return FlaxIsotonicCalibrationPredictor(base, config)
    if config.method == "dirichlet":
        return FlaxDirichletCalibrationPredictor(base, config)
    return FlaxAffineLogitCalibrationPredictor(base, config)
