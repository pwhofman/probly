"""Credal-bounds transformation."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import CredalBoundsPredictor, credal_bounds, credal_bounds_generator


## Torch
@credal_bounds_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "CredalBoundsPredictor",
    "credal_bounds",
    "credal_bounds_generator",
]
