"""Metrics with backend dispatch for NumPy, PyTorch, and JAX."""

from __future__ import annotations

from probly.lazy_types import JAX_ARRAY, JAX_ARRAY_LIKE, TORCH_TENSOR, TORCH_TENSOR_LIKE

from . import array as array

# eagerly register numpy (always available)
from ._common import (
    auc,
    average_interval_width,
    average_precision_score,
    coverage,
    efficiency,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@auc.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@average_interval_width.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@average_precision_score.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@coverage.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@efficiency.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@precision_recall_curve.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@roc_auc_score.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@roc_curve.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@auc.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@average_precision_score.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@precision_recall_curve.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@roc_auc_score.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@roc_curve.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415


__all__ = [
    "auc",
    "average_interval_width",
    "average_precision_score",
    "coverage",
    "efficiency",
    "precision_recall_curve",
    "roc_auc_score",
    "roc_curve",
]
