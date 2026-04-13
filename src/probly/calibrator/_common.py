"""Protocols and ABCs for representation wrappers."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from lazy_dispatch import lazydispatch
from lazy_dispatch.registry_meta import ProtocolRegistry
from probly.utils.switchdispatch import switch

type CalibratorName = Literal[
    "conformal_calibrator",
    "temperature_scaling_calibrator",
    "isotonic_regression_calibrator",
]


@runtime_checkable
class Calibrator[**In, Out](ProtocolRegistry, Protocol, structural_checking=False):
    """Protocol for generic calibrators."""

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        calibrate_method = getattr(subclass, "calibrate", None)
        if calibrate_method is not None and callable(calibrate_method):
            return True
        return NotImplemented


# Protocols for predictors


@runtime_checkable
class ConformalCalibrator[**In, Out](ProtocolRegistry, Protocol, structural_checking=False):
    """Protocol for generic predictors."""

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        calibrate_method = getattr(subclass, "calibrate", None)
        conformal_method = getattr(subclass, "conformal_quantile", None)
        if calibrate_method is not None or conformal_method is not None:
            return True
        if issubclass(subclass, ConformalCalibrator):
            return True
        return NotImplemented


calibrator_registry = switch[CalibratorName, type[Calibrator]]()


@lazydispatch
def calibrate_raw[**In, Out](predictor: Calibrator[In, Out], /, *args: In.args, **kwargs: In.kwargs) -> Out:
    """Calibrate the predictor with the given arguments, returning the raw output of the calibrator."""
    msg = "Calibration not implemented for this type of predictor."
    raise NotImplementedError(msg)


@lazydispatch
def calibrate[**In, Out](predictor: Calibrator[In, Out], /, *args: In.args, **kwargs: In.kwargs) -> Out:
    return calibrate_raw(predictor, *args, **kwargs)
