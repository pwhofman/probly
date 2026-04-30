"""Laplace approximation method (wraps the laplace-torch package)."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import (
    LaplaceGLMPredictor,
    LaplaceMCPredictor,
    LaplacePredictor,
    laplace,
    laplace_generator,
)


@laplace_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "LaplaceGLMPredictor",
    "LaplaceMCPredictor",
    "LaplacePredictor",
    "laplace",
]
