"""Metrics with backend dispatch for NumPy, PyTorch, and JAX."""

from __future__ import annotations

from probly.lazy_types import JAX_VALUE, TORCH_VALUE

from . import numpy as numpy

# eagerly register numpy (always available)
from ._common import (
    ScoringRule,
    _asarray_like,
    _expected_loss,
    _floating,
    _label_values,
    _log_probabilities,
    _machine_epsilon,
    _one_hot_argmax,
    _probabilities,
    accuracy,
    auc,
    average_interval_width,
    average_precision_score,
    brier_loss,
    brier_loss_per_instance,
    classwise_ece,
    convex_hull_coverage,
    coverage,
    efficiency,
    expected_calibration_error,
    false_negative_rate,
    false_positive_rate,
    log_loss,
    log_loss_per_instance,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    spherical_loss,
    spherical_loss_per_instance,
    zero_one_loss,
    zero_one_loss_per_instance,
)


@log_loss_per_instance.delayed_register(TORCH_VALUE)
@_asarray_like.delayed_register(TORCH_VALUE)
@_expected_loss.delayed_register(TORCH_VALUE)
@_floating.delayed_register(TORCH_VALUE)
@_label_values.delayed_register(TORCH_VALUE)
@_log_probabilities.delayed_register(TORCH_VALUE)
@_machine_epsilon.delayed_register(TORCH_VALUE)
@_one_hot_argmax.delayed_register(TORCH_VALUE)
@_probabilities.delayed_register(TORCH_VALUE)
@classwise_ece.delayed_register(TORCH_VALUE)
@convex_hull_coverage.delayed_register(TORCH_VALUE)
@expected_calibration_error.delayed_register(TORCH_VALUE)
@precision_recall_curve.delayed_register(TORCH_VALUE)
@roc_curve.delayed_register(TORCH_VALUE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@log_loss_per_instance.delayed_register(JAX_VALUE)
@_asarray_like.delayed_register(JAX_VALUE)
@_expected_loss.delayed_register(JAX_VALUE)
@_floating.delayed_register(JAX_VALUE)
@_label_values.delayed_register(JAX_VALUE)
@_log_probabilities.delayed_register(JAX_VALUE)
@_machine_epsilon.delayed_register(JAX_VALUE)
@_one_hot_argmax.delayed_register(JAX_VALUE)
@_probabilities.delayed_register(JAX_VALUE)
@classwise_ece.delayed_register(JAX_VALUE)
@convex_hull_coverage.delayed_register(JAX_VALUE)
@expected_calibration_error.delayed_register(JAX_VALUE)
@precision_recall_curve.delayed_register(JAX_VALUE)
@roc_curve.delayed_register(JAX_VALUE)
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415


__all__ = [
    "ScoringRule",
    "accuracy",
    "auc",
    "average_interval_width",
    "average_precision_score",
    "brier_loss",
    "brier_loss_per_instance",
    "classwise_ece",
    "convex_hull_coverage",
    "coverage",
    "efficiency",
    "expected_calibration_error",
    "false_negative_rate",
    "false_positive_rate",
    "log_loss",
    "log_loss_per_instance",
    "precision_recall_curve",
    "roc_auc_score",
    "roc_curve",
    "spherical_loss",
    "spherical_loss_per_instance",
    "zero_one_loss",
    "zero_one_loss_per_instance",
]
