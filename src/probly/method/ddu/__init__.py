"""Deep Deterministic Uncertainty (DDU) for classification models."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import DDUPredictor, ddu, ddu_generator


## Torch
@ddu_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "DDUPredictor",
    "ddu",
]
