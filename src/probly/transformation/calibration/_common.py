"""Calibration wrappers and factory functions for logit scaling."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Concatenate, Literal, Protocol, Self, runtime_checkable

from flextype import flexdispatch
from probly.calibrator._common import Calibrator
from probly.predictor import LogitClassifier, Predictor, ProbabilisticClassifier
from probly.transformation.transformation import predictor_transformation

type ScalingMethod = Literal["temperature", "platt", "vector", "isotonic"]


@dataclass(slots=True, frozen=True)
class CalibrationMethodConfig:
    """Configuration for affine logit scaling variants."""

    method: ScalingMethod
    vector_scale: bool = False
    use_bias: bool = False
    num_classes: int | None = None


@runtime_checkable  # ty:ignore[conflicting-metaclass]
class CalibrationPredictor[**In, T](
    Predictor[In, T],
    Calibrator[Concatenate[T, In], T],
    Protocol,
):
    """Protocol for logit calibration wrappers."""

    predictor: Predictor[In, T]

    def calibrate(self, y_calib: T, *calib_args: In.args, **calib_kwargs: In.kwargs) -> Self:
        """Calibrate scaling parameters on calibration data."""
        ...


@CalibrationPredictor.register
class _CalibrationPredictorBase[**In, Out](ABC):
    """Backend-agnostic metadata for affine scaling wrappers."""

    predictor: Predictor[In, Out]
    config: CalibrationMethodConfig

    def __init__(self, predictor: Predictor[In, Out], config: CalibrationMethodConfig) -> None:
        super().__init__()
        self.predictor = predictor
        self.config = config


@flexdispatch
def calibration_generator[**In, Out](
    base: Predictor[In, Out],
    config: CalibrationMethodConfig,
) -> CalibrationPredictor[In, Out]:
    """Generate a backend-specific calibration wrapper."""
    msg = f"No calibration generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@predictor_transformation(permitted_predictor_types=(LogitClassifier,), preserve_predictor_type=True)
@CalibrationPredictor.register_factory
def temperature_scaling[**In, Out](base: Predictor[In, Out]) -> CalibrationPredictor[In, Out]:
    """Create a temperature scaling calibration wrapper."""
    return calibration_generator(
        base,
        config=CalibrationMethodConfig(method="temperature", vector_scale=False, use_bias=False),
    )


@predictor_transformation(permitted_predictor_types=(LogitClassifier,), preserve_predictor_type=True)
@CalibrationPredictor.register_factory
def platt_scaling[**In, Out](base: Predictor[In, Out]) -> CalibrationPredictor[In, Out]:
    """Create a platt scaling calibration wrapper."""
    return calibration_generator(
        base,
        config=CalibrationMethodConfig(method="platt", vector_scale=False, use_bias=True),
    )


@predictor_transformation(permitted_predictor_types=(LogitClassifier,), preserve_predictor_type=True)
@CalibrationPredictor.register_factory
def vector_scaling[**In, Out](
    base: Predictor[In, Out],
    num_classes: int | None = None,
) -> CalibrationPredictor[In, Out]:
    """Create a vector scaling calibration wrapper."""
    if num_classes is not None and num_classes <= 1:
        msg = f"vector scaling expects num_classes > 1, but got {num_classes}."
        raise ValueError(msg)
    return calibration_generator(
        base,
        config=CalibrationMethodConfig(method="vector", vector_scale=True, use_bias=True, num_classes=num_classes),
    )


@predictor_transformation(
    permitted_predictor_types=(LogitClassifier, ProbabilisticClassifier), preserve_predictor_type=False
)
@ProbabilisticClassifier.register_factory
@CalibrationPredictor.register_factory
def isotonic_regression[**In, Out](base: Predictor[In, Out]) -> CalibrationPredictor[In, Out]:
    """Create an isotonic regression calibration wrapper for binary logits."""
    return calibration_generator(
        base,
        config=CalibrationMethodConfig(method="isotonic", vector_scale=False, use_bias=False),
    )


__all__ = [
    "CalibrationMethodConfig",
    "CalibrationPredictor",
    "_CalibrationPredictorBase",
    "calibration_generator",
    "isotonic_regression",
    "platt_scaling",
    "temperature_scaling",
    "vector_scaling",
]
