"""Implementation of Isotonic Regression."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from .common import _isotonic_factory, isotonic_regression, register_isotonic_factory

__all__ = ["isotonic_regression", "register_isotonic_factory"]


# Torch
@_isotonic_factory.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch  # noqa: F401, PLC0415
