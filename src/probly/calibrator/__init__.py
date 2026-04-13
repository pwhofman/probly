"""This module contains the Protocols for the calibrator submodule."""

from __future__ import annotations

from ._common import ConformalCalibrator, calibrate

__all__ = [
    "ConformalCalibrator",
    "calibrate",
]
