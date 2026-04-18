"""Conformal Calibration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from lazy_dispatch import lazydispatch
from lazy_dispatch.registry_meta import ProtocolRegistry

if TYPE_CHECKING:
    from probly.method.conformal_credal_set_prediction.scores import NonConformityFunction


@runtime_checkable
class Calibrator[**In, Out](ProtocolRegistry, Protocol, structural_checking=False):
    """Protocol for generic calibrations."""


@runtime_checkable
class ConformalCalibrator[**In, Out](ProtocolRegistry, Protocol, structural_checking=False):
    """Protocol for conformal calibrators."""

    quantile: float
    non_conformity_score: NonConformityFunction


@lazydispatch
def calibrate_raw[**In, Out](predictor: Calibrator[In, Out], /, *args: In.args, **kwargs: In.kwargs) -> Out:
    """Calibrate the predictor with the given arguments, returning the raw output of the calibrator."""
    msg = "Calibration not implemented for this type of predictor."
    raise NotImplementedError(msg)


@lazydispatch
def calibrate[**In, Out](predictor: Calibrator[In, Out], /, *args: In.args, **kwargs: In.kwargs) -> Out:
    """Calibrate the predictor with the given arguments."""
    return calibrate_raw(predictor, *args, **kwargs)


@lazydispatch
def calibrate_raw_conformal[**In, Out](
    predictor: ConformalCalibrator[In, Out],
    non_conformity_score: NonConformityFunction,
    /,
    *args: In.args,
    **kwargs: In.kwargs,
) -> Out:
    """Calibrate a conformal predictor with the given arguments, returning the raw output of the calibrator."""
    msg = "Conformal calibration not implemented for this type of predictor."
    raise NotImplementedError(msg)


@lazydispatch
def calibrate_conformal[**In, Out](
    predictor: ConformalCalibrator[In, Out],
    non_conformity_score: NonConformityFunction,
    /,
    *args: In.args,
    **kwargs: In.kwargs,
) -> Out:
    """Calibrate a conformal predictor with the given arguments."""
    return calibrate_raw_conformal(predictor, non_conformity_score, *args, **kwargs)
