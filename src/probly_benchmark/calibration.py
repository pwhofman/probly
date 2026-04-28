"""Benchmark post-hoc calibration registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch

from probly.calibrator import calibrate
from probly.method.calibration import temperature_scaling, torch_identity_logit_model
from probly.predictor import predict_raw

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from omegaconf import DictConfig


DEFAULT_CALIBRATION = "none"
CALIBRATION_ARTIFACT_TYPE = "post_hoc_calibration"
CALIBRATION_STATE_DICT_KEY = "calibration_state_dict"
SOURCE_ARTIFACT_KEY = "source_artifact"
SOURCE_RUN_ID_KEY = "source_run_id"


@dataclass(frozen=True)
class CalibrationSpec:
    """Benchmark metadata for a post-hoc calibration method.

    Attributes:
        transform: Function that wraps an uncalibrated model with a calibrator.
        supported_methods: Benchmark methods this calibration is currently allowed for.
        state_keys: Calibration state keys expected in saved calibration artifacts.
        metric_extractors: Functions that extract calibration-method-specific metrics.
    """

    transform: Callable[..., torch.nn.Module]
    supported_methods: frozenset[str]
    state_keys: frozenset[str]
    metric_extractors: tuple[Callable[[torch.nn.Module], dict[str, float]], ...] = ()


def _temperature_scaling(model: torch.nn.Module, **_: Any) -> torch.nn.Module:  # noqa: ANN401
    """Wrap a torch model with temperature scaling."""
    return cast("torch.nn.Module", temperature_scaling(model))


def _temperature_metrics(calibrator: torch.nn.Module) -> dict[str, float]:
    """Extract scalar temperature scaling metrics from a fitted calibrator."""
    temperature = getattr(calibrator, "temperature", None)
    if not isinstance(temperature, torch.Tensor):
        return {}
    return {"temperature": float(temperature.reshape(-1)[0].item())}


CALIBRATION_METHODS = {
    "temperature_scaling": CalibrationSpec(
        transform=_temperature_scaling,
        supported_methods=frozenset({"base"}),
        state_keys=frozenset({"_temperature", "_bias", "_is_calibrated"}),
        metric_extractors=(_temperature_metrics,),
    ),
}
"""Registry of post-hoc calibration methods supported by the benchmark."""


def get_calibration_name(cfg: DictConfig | dict) -> str:
    """Return the configured post-hoc calibration name."""
    calibration = cfg.get("calibration", None)
    if not calibration:
        return DEFAULT_CALIBRATION
    return str(calibration.get("name", DEFAULT_CALIBRATION)).lower()


def validate_calibration_config(cfg: DictConfig | dict, *, allow_none: bool = True) -> None:
    """Validate that the configured calibration is supported for the benchmark method."""
    name = get_calibration_name(cfg)
    if name == DEFAULT_CALIBRATION:
        if allow_none:
            return
        msg = "calibration=none cannot be used by calibrate.py; choose a calibration method."
        raise ValueError(msg)

    spec = CALIBRATION_METHODS.get(name)
    if spec is None:
        supported = ", ".join(sorted(CALIBRATION_METHODS))
        msg = f"Unknown calibration method {name!r}; supported calibration methods: {supported}."
        raise ValueError(msg)

    method_name = str(cfg.get("method", {}).get("name"))
    if method_name not in spec.supported_methods:
        supported_methods = ", ".join(sorted(spec.supported_methods))
        msg = (
            f"calibration.name={name!r} is only supported for benchmark methods "
            f"({supported_methods}); got method={method_name!r}."
        )
        raise ValueError(msg)


def get_calibration_spec(cfg: DictConfig | dict) -> CalibrationSpec:
    """Return the configured calibration spec."""
    name = get_calibration_name(cfg)
    if name == DEFAULT_CALIBRATION:
        msg = "calibration=none has no CalibrationSpec."
        raise ValueError(msg)
    spec = CALIBRATION_METHODS.get(name)
    if spec is None:
        supported = ", ".join(sorted(CALIBRATION_METHODS))
        msg = f"Unknown calibration method {name!r}; supported calibration methods: {supported}."
        raise ValueError(msg)
    return spec


def calibration_params(cfg: DictConfig | dict) -> dict[str, Any]:
    """Return calibration method parameters as a plain dictionary."""
    calibration_cfg = cfg.get("calibration", {})
    return dict(calibration_cfg.get("params") or {})


def apply_calibration(model: torch.nn.Module, cfg: DictConfig | dict) -> torch.nn.Module:
    """Wrap a model with the configured calibration method."""
    name = get_calibration_name(cfg)
    if name == DEFAULT_CALIBRATION:
        return model
    spec = get_calibration_spec(cfg)
    return spec.transform(model, **calibration_params(cfg))


def build_logit_calibrator(cfg: DictConfig | dict) -> torch.nn.Module:
    """Build a calibrator around an identity logit model for direct logit fitting."""
    return apply_calibration(torch_identity_logit_model(), cfg)


def fit_logit_calibrator(
    cfg: DictConfig | dict,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.nn.Module:
    """Fit the configured calibration method directly on logits and labels."""
    logit_calibrator = build_logit_calibrator(cfg)
    calibrate(logit_calibrator, targets, logits)
    return logit_calibrator


def predict_calibrated_logits(calibrator: torch.nn.Module, logits: torch.Tensor) -> torch.Tensor:
    """Apply a fitted logit calibrator to logits."""
    calibrated_logits = predict_raw(calibrator, logits)
    if not isinstance(calibrated_logits, torch.Tensor):
        msg = f"Expected calibrated logits to be a torch.Tensor, got {type(calibrated_logits)}."
        raise TypeError(msg)
    return calibrated_logits


def extract_calibration_metrics(cfg: DictConfig | dict, calibrator: torch.nn.Module) -> dict[str, float]:
    """Extract calibration-method-specific metrics from a fitted calibrator."""
    metrics: dict[str, float] = {}
    for extractor in get_calibration_spec(cfg).metric_extractors:
        metrics.update(extractor(calibrator))
    return metrics


def load_calibration_state(
    model: torch.nn.Module,
    cfg: DictConfig | dict,
    state_dict: Mapping[str, Any],
) -> torch.nn.Module:
    """Load calibration-only state into a calibrated model wrapper."""
    spec = get_calibration_spec(cfg)
    missing_state_keys = spec.state_keys.difference(state_dict)
    if missing_state_keys:
        missing = ", ".join(sorted(missing_state_keys))
        msg = f"Calibration artifact is missing required state keys: {missing}."
        raise RuntimeError(msg)

    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.unexpected_keys:
        unexpected = ", ".join(load_result.unexpected_keys)
        msg = f"Calibration artifact contains unexpected state keys: {unexpected}."
        raise RuntimeError(msg)
    return model
