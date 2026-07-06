"""Shared Dirichlet calibration method implementation."""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

from flextype import flexdispatch

from probly.calibrator._common import Calibrator
from probly.predictor import LogitClassifier, Predictor
from probly.transformation.transformation import predictor_transformation

_DEFAULT_REG_LAMBDA = 1e-3


@runtime_checkable  # ty:ignore[conflicting-metaclass]
class DirichletCalibrationPredictor[**In, Out](Predictor[In, Out], Calibrator[In, Out], Protocol):
    """A logit predictor recalibrated by Dirichlet calibration.

    Wraps a base logit classifier and applies a learned linear map on the
    log-probabilities, ``W @ ln(softmax(logits)) + b``, fitted by minimising the
    calibration negative log-likelihood with Off-Diagonal and Intercept
    Regularisation (ODIR).
    """

    predictor: Predictor[In, Out]

    def calibrate(self, y_calib: Out, *calib_args: In.args, **calib_kwargs: In.kwargs) -> Self:
        """Calibrate the Dirichlet map on a held-out calibration split."""
        ...


@flexdispatch
def dirichlet_calibration_generator[**In, Out](
    base: Predictor[In, Out],
    num_classes: int,
    reg_lambda: float,
    reg_mu: float,
) -> DirichletCalibrationPredictor[In, Out]:
    """Generate a backend-specific Dirichlet calibration wrapper."""
    msg = f"No Dirichlet calibration generator is registered for type {type(base)}"
    raise NotImplementedError(msg)


@predictor_transformation(permitted_predictor_types=(LogitClassifier,), preserve_predictor_type=True)
@DirichletCalibrationPredictor.register_factory
def dirichlet_calibration[**In, Out](
    base: Predictor[In, Out],
    num_classes: int | None = None,
    reg_lambda: float = _DEFAULT_REG_LAMBDA,
    reg_mu: float | None = None,
) -> DirichletCalibrationPredictor[In, Out]:
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
    effective_mu = reg_lambda if reg_mu is None else reg_mu
    return dirichlet_calibration_generator(base, num_classes, reg_lambda, effective_mu)


__all__ = [
    "DirichletCalibrationPredictor",
    "dirichlet_calibration",
    "dirichlet_calibration_generator",
]
