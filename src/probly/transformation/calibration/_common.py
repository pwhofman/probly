"""Calibration wrappers and factory functions for logit scaling."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Concatenate, Literal, Protocol, Self, runtime_checkable

from flextype import flexdispatch

from probly.calibrator._common import Calibrator
from probly.predictor import BinaryLogitClassifier, BinaryProbabilisticClassifier, LogitClassifier, Predictor
from probly.transformation.transformation import predictor_transformation

type ScalingMethod = Literal["temperature", "platt", "vector", "isotonic", "dirichlet"]


@dataclass(slots=True, frozen=True)
class CalibrationMethodConfig:
    """Configuration for affine logit scaling variants."""

    method: ScalingMethod
    vector_scale: bool = False
    use_bias: bool = False
    num_classes: int | None = None
    reg_lambda: float = 0.0
    reg_mu: float | None = None


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
    """Create a temperature scaling calibration wrapper based on :cite:`guoOnCalibration2017`."""
    return calibration_generator(
        base,
        config=CalibrationMethodConfig(method="temperature", vector_scale=False, use_bias=False),
    )


@predictor_transformation(permitted_predictor_types=(BinaryLogitClassifier,), preserve_predictor_type=True)
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
    """Create a vector scaling calibration wrapper based on :cite:`guoOnCalibration2017`."""
    if num_classes is not None and num_classes <= 1:
        msg = f"vector scaling expects num_classes > 1, but got {num_classes}."
        raise ValueError(msg)
    return calibration_generator(
        base,
        config=CalibrationMethodConfig(method="vector", vector_scale=True, use_bias=True, num_classes=num_classes),
    )


@predictor_transformation(permitted_predictor_types=(LogitClassifier,), preserve_predictor_type=True)
@CalibrationPredictor.register_factory
def dirichlet_calibration[**In, Out](
    base: Predictor[In, Out],
    num_classes: int | None = None,
    reg_lambda: float = 1e-3,
    reg_mu: float | None = None,
) -> CalibrationPredictor[In, Out]:
    """Turn a multi-class logit classifier into a Dirichlet-calibrated classifier.

    Based on :cite:`kullBeyondTemperatureScaling2019`.  Dirichlet calibration fits
    a multinomial logistic regression on the log-probabilities of the base
    classifier, ``q = softmax(W @ ln(p) + b)`` with a full ``num_classes x
    num_classes`` weight matrix ``W`` and bias ``b``.  This generalises temperature
    and vector scaling and recalibrates probabilities directly rather than logits.

    The returned predictor still needs its parameters fitted on a held-out
    calibration split via :func:`probly.calibrator.calibrate` (or the wrapper's
    ``calibrate``/``fit`` methods) before it can be used for prediction.

    Args:
        base: Base logit classifier to be calibrated.
        num_classes: Number of classes ``k`` the classifier predicts.  Must be
            greater than one and match the class axis of the logits.
        reg_lambda: Strength of the Off-Diagonal Regularisation, an L2 penalty on
            the off-diagonal entries of ``W``.  ``0`` disables it, recovering the
            unregularised full-matrix fit.
        reg_mu: Strength of the Intercept Regularisation, an L2 penalty on the bias
            ``b``.  Defaults to ``reg_lambda`` when ``None`` (the paper's ODIR
            convention of tying the two strengths together).

    Returns:
        The Dirichlet calibration predictor, awaiting calibration.
    """
    if num_classes is None or num_classes <= 1:
        msg = f"Dirichlet calibration expects num_classes > 1, but got {num_classes}."
        raise ValueError(msg)
    return calibration_generator(
        base,
        config=CalibrationMethodConfig(
            method="dirichlet",
            vector_scale=False,
            use_bias=True,
            num_classes=num_classes,
            reg_lambda=reg_lambda,
            reg_mu=reg_lambda if reg_mu is None else reg_mu,
        ),
    )


@predictor_transformation(
    permitted_predictor_types=(BinaryLogitClassifier, BinaryProbabilisticClassifier), preserve_predictor_type=False
)
@BinaryProbabilisticClassifier.register_factory
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
    "dirichlet_calibration",
    "isotonic_regression",
    "platt_scaling",
    "temperature_scaling",
    "vector_scaling",
]
