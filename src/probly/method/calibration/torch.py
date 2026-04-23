"""PyTorch logit calibration wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Self, override

import torch
from torch import nn
import torch.nn.functional as F

from probly.predictor import LogitClassifier, predict_raw

from ._common import CalibrationMethodConfig, _CalibrationPredictorBase, calibration_generator

_EPS = 1e-6
_LBFGS_MAX_ITER = 128
_ISOTONIC_MAX_KNOTS = 4096


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value))


def _reshape_binary_preds(preds: torch.Tensor) -> torch.Tensor:
    if preds.ndim >= 2 and preds.shape[-1] == 1:
        return preds.squeeze(-1)
    return preds


def _reshape_binary_labels(labels: torch.Tensor, expected_elements: int) -> torch.Tensor:
    flat_labels = labels.reshape(-1)
    if flat_labels.numel() != expected_elements:
        msg = (
            "Binary calibration labels must match logits batch size. "
            f"Got {flat_labels.numel()} labels for {expected_elements} logits."
        )
        raise ValueError(msg)
    return flat_labels


def _reshape_multiclass_labels(labels: torch.Tensor, batch_shape: tuple[int, ...]) -> torch.Tensor:
    expected_elements = math.prod(batch_shape) if batch_shape else 1
    flat_labels = labels.reshape(-1)
    if flat_labels.numel() != expected_elements:
        msg = (
            "Multiclass calibration labels must match logits batch size. "
            f"Got {flat_labels.numel()} labels for {expected_elements} logits."
        )
        raise ValueError(msg)
    return flat_labels


def _calibration_loss(scaled_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    maybe_multiclass = scaled_logits.ndim >= 2 and scaled_logits.shape[-1] > 1
    batch_elements = math.prod(tuple(scaled_logits.shape[:-1])) if scaled_logits.ndim > 1 else 1
    labels_match_batch = labels.reshape(-1).numel() == batch_elements
    single_sample_multiclass = scaled_logits.ndim == 1 and scaled_logits.numel() > 1 and labels.reshape(-1).numel() == 1
    if (maybe_multiclass and labels_match_batch) or single_sample_multiclass:
        num_classes = int(scaled_logits.shape[-1])
        logits_2d = scaled_logits.reshape(-1, num_classes)
        labels_flat = _reshape_multiclass_labels(labels, tuple(scaled_logits.shape[:-1]))
        return F.cross_entropy(logits_2d, labels_flat.long())

    binary_logits = _reshape_binary_preds(scaled_logits).reshape(-1)
    labels_flat = _reshape_binary_labels(labels, binary_logits.numel())
    targets = labels_flat.to(dtype=binary_logits.dtype)
    return F.binary_cross_entropy_with_logits(binary_logits, targets)


def _prepare_binary_isotonic_inputs(
    preds: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, bool]:
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
    if flat_labels.numel() != flat_logits.numel():
        msg = (
            "Binary isotonic calibration labels must match logits batch size. "
            f"Got {flat_labels.numel()} labels for {flat_logits.numel()} logits."
        )
        raise ValueError(msg)
    return flat_logits, flat_labels, has_singleton_class_axis


def _apply_affine(logits: torch.Tensor, temperature: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    if logits.ndim < 1:
        msg = f"Expected logits with at least one dimension, got shape {tuple(logits.shape)}."
        raise ValueError(msg)

    if temperature.numel() == 1:
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
    if bias.numel() == 1:
        return scaled + bias.reshape(())
    if logits.ndim < 2 or bias.shape[-1] != logits.shape[-1]:
        msg = (
            "Bias parameters do not match logits class dimension: "
            f"{bias.shape[-1]} vs {logits.shape[-1] if logits.ndim >= 1 else 'unknown'}."
        )
        raise ValueError(msg)
    return scaled + bias


@LogitClassifier.register_factory
class TorchIdentityLogitModel(nn.Module):
    """Pass-through torch model returning provided logits unchanged."""

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Return input logits unchanged."""
        return logits


class _TorchCalibrationPredictorBase[**In](_CalibrationPredictorBase[In, torch.Tensor], nn.Module, ABC):
    predictor: nn.Module

    @abstractmethod
    def calibrate(self, y_calib: torch.Tensor, *calib_args: In.args, **calib_kwargs: In.kwargs) -> Self:
        """Calibrate model parameters on a held-out calibration split."""
        raise NotImplementedError

    @property
    def is_calibrated(self) -> bool:
        """Return whether calibration parameters were fitted."""
        value = self._buffers.get("_is_calibrated")
        return isinstance(value, torch.Tensor) and bool(value.item())

    def fit(self, x_calib: torch.Tensor, y_calib: torch.Tensor) -> Self:
        """Fit alias mapping sklearn-style argument order to calibrate."""
        return self.calibrate(y_calib, x_calib)  # ty:ignore[invalid-argument-type]


class TorchAffineLogitCalibrationPredictor[**In](_TorchCalibrationPredictorBase[In]):
    """Torch wrapper for parametric affine logit calibration methods."""

    def __init__(self, predictor: nn.Module, config: CalibrationMethodConfig) -> None:
        """Initialize affine calibration buffers for the selected method."""
        super().__init__(predictor, config)
        self.register_buffer("_temperature", self._initial_temperature_buffer(config))
        self.register_buffer("_bias", self._initial_bias_buffer(config))
        self.register_buffer("_is_calibrated", torch.tensor(False, dtype=torch.bool))

    @staticmethod
    def _initial_temperature_buffer(config: CalibrationMethodConfig) -> torch.Tensor:
        if config.vector_scale and config.num_classes is not None:
            return torch.full((config.num_classes,), float("nan"), dtype=torch.float64)
        return torch.tensor(float("nan"), dtype=torch.float64)

    @staticmethod
    def _initial_bias_buffer(config: CalibrationMethodConfig) -> torch.Tensor:
        if not config.use_bias:
            return torch.tensor(float("nan"), dtype=torch.float64)
        if config.vector_scale and config.num_classes is not None:
            return torch.full((config.num_classes,), float("nan"), dtype=torch.float64)
        return torch.tensor(float("nan"), dtype=torch.float64)

    @property
    def temperature(self) -> torch.Tensor | None:
        """Return fitted temperature parameters when available."""
        if not self.is_calibrated:
            return None
        temperature = self._buffers.get("_temperature")
        if not isinstance(temperature, torch.Tensor):
            return None
        return temperature.detach().clone()

    @property
    def bias(self) -> torch.Tensor | None:
        """Return fitted bias parameters when available."""
        if not self.config.use_bias or not self.is_calibrated:
            return None
        bias = self._buffers.get("_bias")
        if not isinstance(bias, torch.Tensor):
            return None
        return bias.detach().clone()

    def _require_calibrated(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.is_calibrated:
            msg = "Calibration wrapper is not calibrated. Please call calibrate() before prediction."
            raise ValueError(msg)

        temperature = self._buffers.get("_temperature")
        if not isinstance(temperature, torch.Tensor):
            msg = "Calibrated temperature buffer is missing."
            raise ValueError(msg)  # noqa: TRY004

        if not self.config.use_bias:
            return temperature, None

        bias = self._buffers.get("_bias")
        if not isinstance(bias, torch.Tensor):
            msg = "Calibrated bias buffer is missing."
            raise ValueError(msg)  # noqa: TRY004
        return temperature, bias

    def _validate_vector_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> int:
        if logits.ndim < 2 or logits.shape[-1] <= 1:
            msg = "Vector scaling expects logits with an explicit class axis and more than one class."
            raise ValueError(msg)
        num_classes = int(logits.shape[-1])
        expected_elements = math.prod(tuple(logits.shape[:-1])) if logits.ndim > 1 else 1
        if labels.reshape(-1).numel() != expected_elements:
            msg = (
                "Vector scaling labels must match logits batch size. "
                f"Got {labels.reshape(-1).numel()} labels for {expected_elements} logits."
            )
            raise ValueError(msg)
        if self.config.num_classes is not None and self.config.num_classes != num_classes:
            msg = f"Expected logits with {self.config.num_classes} classes, but got logits with {num_classes} classes."
            raise ValueError(msg)
        return num_classes

    def _initial_temperature(self, num_classes: int, logits: torch.Tensor) -> torch.Tensor:
        if self.config.vector_scale:
            return logits.new_ones((num_classes,))
        return logits.new_ones(())

    def _initial_bias(self, num_classes: int, logits: torch.Tensor) -> torch.Tensor:
        if self.config.vector_scale:
            return logits.new_zeros((num_classes,))
        return logits.new_zeros(())

    @override
    def calibrate(self, y_calib: torch.Tensor, *calib_args: In.args, **calib_kwargs: In.kwargs) -> Self:
        """Calibrate affine scaling parameters on calibration data."""
        raw_logits = predict_raw(self.predictor, *calib_args, **calib_kwargs)
        if not isinstance(raw_logits, torch.Tensor):
            msg = f"Torch calibration expects torch logits, got {type(raw_logits)}"
            raise TypeError(msg)

        logits = raw_logits.detach()
        labels = y_calib if isinstance(y_calib, torch.Tensor) else torch.as_tensor(y_calib, device=logits.device)
        labels = labels.to(device=logits.device)

        if logits.ndim < 1:
            msg = f"Expected logits with at least one dimension, got shape {tuple(logits.shape)}."
            raise ValueError(msg)

        num_classes = self._validate_vector_logits(logits, labels) if self.config.vector_scale else 1
        initial_temperature = self._initial_temperature(num_classes, logits)
        raw_temperature = _inverse_softplus(torch.clamp(initial_temperature - _EPS, min=_EPS))
        log_temperature = nn.Parameter(raw_temperature)

        bias_parameter: nn.Parameter | None = None
        parameters: list[nn.Parameter] = [log_temperature]
        if self.config.use_bias:
            bias_parameter = nn.Parameter(self._initial_bias(num_classes, logits))
            parameters.append(bias_parameter)

        optimizer = torch.optim.LBFGS(parameters, max_iter=_LBFGS_MAX_ITER, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            temperature = F.softplus(log_temperature) + _EPS
            scaled_logits = _apply_affine(logits, temperature, bias_parameter)
            loss = _calibration_loss(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            self._temperature = (F.softplus(log_temperature) + _EPS).detach().clone()
            if self.config.use_bias and bias_parameter is not None:
                self._bias = bias_parameter.detach().clone()
            calibrated_flag = self._buffers.get("_is_calibrated")
            if isinstance(calibrated_flag, torch.Tensor):
                calibrated_flag.fill_(True)

        return self

    def forward(self, *args: In.args, **kwargs: In.kwargs) -> torch.Tensor:
        """Apply calibrated affine scaling to model logits."""
        raw_logits = predict_raw(self.predictor, *args, **kwargs)
        if not isinstance(raw_logits, torch.Tensor):
            msg = f"Torch calibration expects torch logits, got {type(raw_logits)}"
            raise TypeError(msg)
        temperature, bias = self._require_calibrated()
        return _apply_affine(raw_logits, temperature, bias)


class TorchIsotonicCalibrationPredictor[**In](_TorchCalibrationPredictorBase[In]):
    """Torch wrapper for non-parametric isotonic calibration on binary logits."""

    def __init__(self, predictor: nn.Module, config: CalibrationMethodConfig) -> None:
        """Initialize fixed-layout buffers for isotonic knot serialization."""
        super().__init__(predictor, config)
        self.register_buffer("_isotonic_x_knots", torch.full((_ISOTONIC_MAX_KNOTS,), float("nan"), dtype=torch.float64))
        self.register_buffer("_isotonic_y_knots", torch.full((_ISOTONIC_MAX_KNOTS,), float("nan"), dtype=torch.float64))
        self.register_buffer("_isotonic_num_knots", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("_is_calibrated", torch.tensor(False, dtype=torch.bool))

    @staticmethod
    def _fit_isotonic_knots(preds: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415
        except ImportError as err:  # pragma: no cover - guarded by optional dependency
            msg = "Torch isotonic calibration requires scikit-learn to fit isotonic regression knots."
            raise ImportError(msg) from err

        x_np = preds.detach().cpu().numpy()
        y_np = labels.detach().cpu().numpy().astype(float)
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(x_np, y_np)

        x_knots = torch.as_tensor(model.X_thresholds_, dtype=torch.float64)
        y_knots = torch.as_tensor(model.y_thresholds_, dtype=torch.float64)
        return x_knots, y_knots

    def _store_isotonic_knots(self, x_knots: torch.Tensor, y_knots: torch.Tensor) -> None:
        if x_knots.numel() > _ISOTONIC_MAX_KNOTS:
            msg = (
                "Isotonic calibration produced more knots than supported by the fixed state_dict layout: "
                f"{x_knots.numel()} > {_ISOTONIC_MAX_KNOTS}."
            )
            raise ValueError(msg)

        with torch.no_grad():
            x_buffer = self._buffers.get("_isotonic_x_knots")
            y_buffer = self._buffers.get("_isotonic_y_knots")
            n_buffer = self._buffers.get("_isotonic_num_knots")
            if (
                not isinstance(x_buffer, torch.Tensor)
                or not isinstance(y_buffer, torch.Tensor)
                or not isinstance(n_buffer, torch.Tensor)
            ):
                msg = "Isotonic calibration buffers are missing."
                raise TypeError(msg)

            x_buffer.fill_(float("nan"))
            y_buffer.fill_(float("nan"))
            x_buffer[: x_knots.numel()] = x_knots
            y_buffer[: y_knots.numel()] = y_knots
            n_buffer.fill_(int(x_knots.numel()))
            calibrated_flag = self._buffers.get("_is_calibrated")
            if isinstance(calibrated_flag, torch.Tensor):
                calibrated_flag.fill_(True)

    def _require_isotonic_knots(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_calibrated:
            msg = "Calibration wrapper is not calibrated. Please call calibrate() before prediction."
            raise ValueError(msg)
        num_knots_tensor = self._buffers.get("_isotonic_num_knots")
        x_knots = self._buffers.get("_isotonic_x_knots")
        y_knots = self._buffers.get("_isotonic_y_knots")
        if (
            not isinstance(num_knots_tensor, torch.Tensor)
            or not isinstance(x_knots, torch.Tensor)
            or not isinstance(y_knots, torch.Tensor)
        ):
            msg = "Isotonic calibration buffers are missing."
            raise TypeError(msg)

        num_knots = int(num_knots_tensor.item())
        if num_knots <= 0:
            msg = "Isotonic calibration has no fitted knots."
            raise ValueError(msg)
        return x_knots[:num_knots], y_knots[:num_knots]

    def _apply_isotonic(self, preds: torch.Tensor) -> torch.Tensor:
        flat_preds, _, had_singleton_axis = _prepare_binary_isotonic_inputs(preds, torch.zeros_like(preds))
        x_knots, y_knots = self._require_isotonic_knots()

        x_knots = x_knots.to(device=flat_preds.device, dtype=flat_preds.dtype)
        y_knots = y_knots.to(device=flat_preds.device, dtype=flat_preds.dtype)

        if x_knots.numel() == 1:
            probs = y_knots.expand_as(flat_preds)
        else:
            interior = x_knots[1:-1]
            interval_idx = torch.bucketize(flat_preds, interior, right=False)
            left_x = x_knots[interval_idx]
            right_x = x_knots[interval_idx + 1]
            left_y = y_knots[interval_idx]
            right_y = y_knots[interval_idx + 1]
            denominator = torch.clamp(right_x - left_x, min=1e-12)
            weight = torch.clamp((flat_preds - left_x) / denominator, min=0.0, max=1.0)
            probs = left_y + weight * (right_y - left_y)

        calibrated_probs = probs.reshape(_reshape_binary_preds(preds).shape)
        if had_singleton_axis:
            return calibrated_probs.unsqueeze(-1)
        return calibrated_probs

    @override
    def calibrate(self, y_calib: torch.Tensor, *calib_args: In.args, **calib_kwargs: In.kwargs) -> Self:
        """Calibrate isotonic parameters on calibration data."""
        raw_preds = predict_raw(self.predictor, *calib_args, **calib_kwargs)
        if not isinstance(raw_preds, torch.Tensor):
            msg = f"Torch calibration expects torch predictions, got {type(raw_preds)}"
            raise TypeError(msg)

        preds = raw_preds.detach()
        labels = y_calib if isinstance(y_calib, torch.Tensor) else torch.as_tensor(y_calib, device=preds.device)
        labels = labels.to(device=preds.device)

        flat_preds, flat_labels, _ = _prepare_binary_isotonic_inputs(preds, labels)
        x_knots, y_knots = self._fit_isotonic_knots(flat_preds, flat_labels)
        self._store_isotonic_knots(x_knots, y_knots)
        return self

    def forward(self, *args: In.args, **kwargs: In.kwargs) -> torch.Tensor:
        """Apply fitted isotonic calibration to model logits."""
        raw_preds = predict_raw(self.predictor, *args, **kwargs)
        if not isinstance(raw_preds, torch.Tensor):
            msg = f"Torch calibration expects torch logits, got {type(raw_preds)}"
            raise TypeError(msg)
        return self._apply_isotonic(raw_preds)


@calibration_generator.register(nn.Module)
def generate_torch_scaling_calibrator(
    base: nn.Module, config: CalibrationMethodConfig
) -> _TorchCalibrationPredictorBase:
    """Create torch calibration wrappers from method configuration."""
    if config.method == "isotonic":
        return TorchIsotonicCalibrationPredictor(base, config)
    return TorchAffineLogitCalibrationPredictor(base, config)
