"""PyTorch implementation of Metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from probly.metrics import (
    auc,
    average_precision_score,
    coverage,
    efficiency,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from probly.representation.conformal_set.torch import TorchIntervalConformalSet, TorchOneHotConformalSet
from probly.representation.credal_set.torch import TorchConvexCredalSet, TorchProbabilityIntervalsCredalSet

if TYPE_CHECKING:
    from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution


@auc.register(torch.Tensor)
def auc_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute area under a curve using the trapezoid rule."""
    return torch.trapezoid(y, x, dim=-1)


@average_precision_score.register(torch.Tensor)
def average_precision_score_torch(y_true: torch.Tensor, y_score: torch.Tensor) -> torch.Tensor:
    """Compute average precision for PyTorch tensors."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return -torch.sum(torch.diff(recall, dim=-1) * precision[..., :-1], dim=-1)  # ty:ignore[invalid-argument-type, not-subscriptable]


@precision_recall_curve.register(torch.Tensor)
def precision_recall_curve_torch(
    y_true: torch.Tensor, y_score: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute precision-recall curve along the last axis."""
    y_true = y_true.float()
    y_score = y_score.float()
    n = y_score.shape[-1]

    desc_idx = torch.argsort(y_score, dim=-1, stable=True, descending=True)
    y_score_sorted = torch.take_along_dim(y_score, desc_idx, dim=-1)
    y_true_sorted = torch.take_along_dim(y_true, desc_idx, dim=-1)

    tps = torch.cumsum(y_true_sorted, dim=-1)
    predicted_pos = torch.arange(1, n + 1, device=y_true.device, dtype=y_true.dtype)
    total_pos = tps[..., -1:]

    precision = tps / predicted_pos
    recall = torch.where(
        total_pos > 0, tps / torch.where(total_pos > 0, total_pos, torch.ones_like(total_pos)), torch.zeros_like(tps)
    )

    ones = torch.ones((*y_score.shape[:-1], 1), device=y_true.device, dtype=y_true.dtype)
    zeros = torch.zeros((*y_score.shape[:-1], 1), device=y_true.device, dtype=y_true.dtype)
    precision = torch.cat([precision.flip(-1), ones], dim=-1)
    recall = torch.cat([recall.flip(-1), zeros], dim=-1)

    return precision, recall, y_score_sorted


@roc_auc_score.register(torch.Tensor)
def roc_auc_score_torch(y_true: torch.Tensor, y_score: torch.Tensor) -> torch.Tensor:
    """Compute area under the ROC curve for PyTorch tensors."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)  # ty:ignore[invalid-return-type]


@roc_curve.register(torch.Tensor)
def roc_curve_torch(y_true: torch.Tensor, y_score: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ROC curve along the last axis."""
    y_true = y_true.float()
    y_score = y_score.float()
    n = y_score.shape[-1]

    desc_idx = torch.argsort(y_score, dim=-1, stable=True, descending=True)
    y_score_sorted = torch.take_along_dim(y_score, desc_idx, dim=-1)
    y_true_sorted = torch.take_along_dim(y_true, desc_idx, dim=-1)

    tps = torch.cumsum(y_true_sorted, dim=-1)
    fps = torch.arange(1, n + 1, device=y_true.device, dtype=y_true.dtype) - tps

    total_pos = tps[..., -1:]
    total_neg = fps[..., -1:]

    tpr = torch.where(
        total_pos > 0, tps / torch.where(total_pos > 0, total_pos, torch.ones_like(total_pos)), torch.zeros_like(tps)
    )
    fpr = torch.where(
        total_neg > 0, fps / torch.where(total_neg > 0, total_neg, torch.ones_like(total_neg)), torch.zeros_like(fps)
    )

    zeros = torch.zeros((*y_score.shape[:-1], 1), device=y_true.device, dtype=y_true.dtype)
    tpr = torch.cat([zeros, tpr], dim=-1)
    fpr = torch.cat([zeros, fpr], dim=-1)
    thresholds = torch.cat([y_score_sorted[..., :1] + 1, y_score_sorted], dim=-1)

    return fpr, tpr, thresholds


# --- Predicted-set metrics ----------------------------------------------------


def _interval_dominance_mask(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Boolean mask of classes whose upper probability exceeds the global lower max."""
    threshold, _ = torch.max(lower, dim=-1, keepdim=True)
    return upper >= threshold


@coverage.register(TorchOneHotConformalSet)
def _coverage_torch_one_hot_conformal_set(y_pred: TorchOneHotConformalSet, y_true: torch.Tensor) -> float:
    """Coverage for a one-hot conformal set: true class is in the selected set."""
    mask = y_pred.tensor
    indices = torch.as_tensor(y_true, device=mask.device).to(dtype=torch.int64).unsqueeze(-1)
    return float(torch.gather(mask, -1, indices).squeeze(-1).float().mean().item())


@efficiency.register(TorchOneHotConformalSet)
def _efficiency_torch_one_hot_conformal_set(y_pred: TorchOneHotConformalSet) -> float:
    """Average cardinality of a one-hot conformal set."""
    return float(y_pred.tensor.float().sum(dim=-1).mean().item())


@coverage.register(TorchIntervalConformalSet)
def _coverage_torch_interval_conformal_set(y_pred: TorchIntervalConformalSet, y_true: torch.Tensor) -> float:
    """Coverage for an interval conformal set: ``lower <= y_true <= upper``."""
    arr = y_pred.tensor
    y = torch.as_tensor(y_true, device=arr.device)
    inside = (y >= arr[..., 0]) & (y <= arr[..., 1])
    return float(inside.float().mean().item())


@efficiency.register(TorchIntervalConformalSet)
def _efficiency_torch_interval_conformal_set(y_pred: TorchIntervalConformalSet) -> float:
    """Average width ``upper - lower`` of an interval conformal set."""
    return float((y_pred.tensor[..., 1] - y_pred.tensor[..., 0]).float().mean().item())


@coverage.register(TorchConvexCredalSet)
def _coverage_torch_convex_credal_set(y_pred: TorchConvexCredalSet, y_true: TorchCategoricalDistribution) -> float:
    """Coverage for a convex credal set: target lies in the convex hull of the vertices.

    ``scipy.optimize.linprog`` is numpy-only, so vertices and targets are
    detached, moved to CPU, and dispatched through the numpy LP path.
    """
    from probly.metrics.array import _coverage_array_convex_credal_set  # noqa: PLC0415
    from probly.representation.credal_set.array import ArrayConvexCredalSet  # noqa: PLC0415
    from probly.representation.distribution.array_categorical import (  # noqa: PLC0415
        ArrayProbabilityCategoricalDistribution,
    )

    vertex_probs = y_pred.tensor.probabilities.detach().cpu().numpy()
    target_probs = y_true.probabilities.detach().cpu().numpy()
    np_cs = ArrayConvexCredalSet(array=ArrayProbabilityCategoricalDistribution(vertex_probs))
    np_target = ArrayProbabilityCategoricalDistribution(target_probs)
    return float(_coverage_array_convex_credal_set(np_cs, np_target))


@efficiency.register(TorchConvexCredalSet)
def _efficiency_torch_convex_credal_set(y_pred: TorchConvexCredalSet) -> float:
    """Cardinality of the interval-dominance prediction set built from the vertex envelope."""
    mask = _interval_dominance_mask(y_pred.lower(), y_pred.upper())
    return float(mask.float().sum(dim=-1).mean().item())


@coverage.register(TorchProbabilityIntervalsCredalSet)
def _coverage_torch_probability_intervals_credal_set(
    y_pred: TorchProbabilityIntervalsCredalSet,
    y_true: TorchCategoricalDistribution,
) -> float:
    """Coverage for a probability-intervals credal set: ``lower[k] <= target[k] <= upper[k]`` for all ``k``."""
    target = y_true.probabilities.to(device=y_pred.lower_bounds.device)
    contained = y_pred.contains(target)
    return float(contained.float().mean().item())


@efficiency.register(TorchProbabilityIntervalsCredalSet)
def _efficiency_torch_probability_intervals_credal_set(y_pred: TorchProbabilityIntervalsCredalSet) -> float:
    """Cardinality of the interval-dominance prediction set."""
    mask = _interval_dominance_mask(y_pred.lower(), y_pred.upper())
    return float(mask.float().sum(dim=-1).mean().item())
