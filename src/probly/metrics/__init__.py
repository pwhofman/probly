"""Metrics with backend dispatch for NumPy, PyTorch, and JAX."""

from __future__ import annotations

from probly.lazy_types import JAX_ARRAY, TORCH_TENSOR

# eagerly register numpy (always available)
from . import (
    numpy_auc as numpy_auc,
    numpy_average_precision_score as numpy_average_precision_score,
    numpy_precision_recall_curve as numpy_precision_recall_curve,
    numpy_roc_auc_score as numpy_roc_auc_score,
    numpy_roc_curve as numpy_roc_curve,
)
from ._common import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


## Torch
@auc.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch_auc as torch_auc  # noqa: PLC0415


@roc_curve.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch_roc_curve as torch_roc_curve  # noqa: PLC0415


@precision_recall_curve.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch_precision_recall_curve as torch_precision_recall_curve  # noqa: PLC0415


@roc_auc_score.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch_roc_auc_score as torch_roc_auc_score  # noqa: PLC0415


@average_precision_score.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch_average_precision_score as torch_average_precision_score  # noqa: PLC0415


## JAX
@auc.delayed_register(JAX_ARRAY)
def _(_: type) -> None:
    from . import jax_auc as jax_auc  # noqa: PLC0415


@roc_curve.delayed_register(JAX_ARRAY)
def _(_: type) -> None:
    from . import jax_roc_curve as jax_roc_curve  # noqa: PLC0415


@precision_recall_curve.delayed_register(JAX_ARRAY)
def _(_: type) -> None:
    from . import jax_precision_recall_curve as jax_precision_recall_curve  # noqa: PLC0415


@roc_auc_score.delayed_register(JAX_ARRAY)
def _(_: type) -> None:
    from . import jax_roc_auc_score as jax_roc_auc_score  # noqa: PLC0415


@average_precision_score.delayed_register(JAX_ARRAY)
def _(_: type) -> None:
    from . import jax_average_precision_score as jax_average_precision_score  # noqa: PLC0415


__all__ = [
    "auc",
    "average_precision_score",
    "precision_recall_curve",
    "roc_auc_score",
    "roc_curve",
]
