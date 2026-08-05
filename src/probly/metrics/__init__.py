"""Metrics with backend dispatch for NumPy, PyTorch, and JAX."""

from __future__ import annotations

from probly.lazy_types import JAX_ARRAY, JAX_ARRAY_LIKE, TORCH_TENSOR, TORCH_TENSOR_LIKE

from . import array as array

# eagerly register numpy (always available)
from ._common import (
    accuracy,
    auc,
    average_interval_width,
    average_precision_score,
    classwise_ece,
    convex_hull_coverage,
    coverage,
    efficiency,
    expected_calibration_error,
    false_negative_rate,
    false_positive_rate,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@accuracy.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@auc.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@average_interval_width.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@average_precision_score.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@classwise_ece.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@convex_hull_coverage.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@coverage.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@efficiency.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@expected_calibration_error.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@false_negative_rate.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@false_positive_rate.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@precision_recall_curve.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@roc_auc_score.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@roc_curve.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@accuracy.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@auc.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@average_precision_score.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@classwise_ece.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@expected_calibration_error.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@false_negative_rate.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@false_positive_rate.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@precision_recall_curve.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@roc_auc_score.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
@roc_curve.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415


__all__ = [
    "accuracy",
    "auc",
    "average_interval_width",
    "average_precision_score",
    "classwise_ece",
    "convex_hull_coverage",
    "coverage",
    "efficiency",
    "expected_calibration_error",
    "false_negative_rate",
    "false_positive_rate",
    "precision_recall_curve",
    "roc_auc_score",
    "roc_curve",
]
