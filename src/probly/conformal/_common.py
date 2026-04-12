"""This module contains common utilities for conformal nonconformity scores."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from probly.calibrator._common import Calibrator


@runtime_checkable
class ConformalCalibrator[**In, Out](Calibrator[In, Out], Protocol):
    """Protocol for split conformal calibrators."""

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        calibrate_method = getattr(subclass, "predict_conformal", None)
        if calibrate_method is not None:
            return True
        if callable(subclass):
            return True
        return NotImplemented

