"""Selective prediction evaluation task with backend dispatch for NumPy, PyTorch, and JAX."""

from __future__ import annotations

from probly.lazy_types import JAX_VALUE, TORCH_VALUE

from . import numpy as numpy
from ._common import evaluate_selective_prediction, selective_prediction


@selective_prediction.delayed_register(TORCH_VALUE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@selective_prediction.delayed_register(JAX_VALUE)
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415


__all__ = [
    "evaluate_selective_prediction",
    "selective_prediction",
]
