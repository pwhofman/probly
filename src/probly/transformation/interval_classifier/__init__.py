"""Interval classifier transformation for uncertainty quantification."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import IntervalClassifierPredictor, interval_classifier, interval_classifier_traverser


## Torch
@interval_classifier_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "IntervalClassifierPredictor",
    "interval_classifier",
    "interval_classifier_traverser",
]
