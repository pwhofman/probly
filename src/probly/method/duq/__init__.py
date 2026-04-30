"""Deterministic Uncertainty Quantification (DUQ) for classification models."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import DUQPredictor, duq, duq_generator


@duq_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "DUQPredictor",
    "duq",
]
