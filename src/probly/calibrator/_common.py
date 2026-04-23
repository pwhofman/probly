"""Protocols and ABCs for representation wrappers."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from flextype import flexdispatch
from flextype.registry_meta import ProtocolRegistry

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


@flexdispatch
def calibrate[**In, Out](
    predictor: Calibrator[In, Out],
    alpha: float,
    y_calib: Out,
    /,
    *calib_args: In.args,
    **calib_kwargs: In.kwargs,
) -> None:
    """Calibrate the predictor with the given arguments."""
    if hasattr(predictor, "calibrate"):
        return predictor.calibrate(alpha, y_calib, *calib_args, **calib_kwargs)  # ty:ignore[call-non-callable]
    msg = "Conformal calibration not implemented for this type of predictor."
    raise NotImplementedError(msg)
