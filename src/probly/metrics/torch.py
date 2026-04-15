"""PyTorch implementation of Metrics."""

from __future__ import annotations

import torch

from probly.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from probly.metrics._common import (
    average_interval_size,
    average_set_size,
    empirical_coverage_classification,
    empirical_coverage_regression,
)
from probly.representation.conformal_set.torch import TorchIntervalConformalSet, TorchOneHotConformalSet


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


@empirical_coverage_classification.register(torch.Tensor)
def _empirical_coverage_classification_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    contained = y_pred[torch.arange(len(y_true)), y_true.long()]
    return contained.float().mean().cpu().item()


@empirical_coverage_regression.register(torch.Tensor)
def _empirical_coverage_regression_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return ((y_true >= y_pred[:, 0]) & (y_true <= y_pred[:, 1])).float().mean().cpu().item()


@average_set_size.register(torch.Tensor)
def _average_set_size_torch(y_pred: torch.Tensor) -> float:
    return y_pred.sum(dim=1).float().mean().cpu().item()


@average_interval_size.register(torch.Tensor)
def _average_interval_size_torch(y_pred: torch.Tensor) -> float:
    return (y_pred[:, 1] - y_pred[:, 0]).float().mean().cpu().item()


@average_interval_size.register(TorchIntervalConformalSet)
def _average_interval_size_torch_interval(y_pred: TorchIntervalConformalSet) -> float:
    return average_interval_size(y_pred.tensor.cpu())


@average_set_size.register(TorchOneHotConformalSet)
def _average_set_size_torch_onehot(y_pred: TorchOneHotConformalSet) -> float:
    return average_set_size(y_pred.tensor.cpu().numpy())


@empirical_coverage_regression.register(TorchIntervalConformalSet)
def _empirical_coverage_regression_torch_interval[T](y_pred: TorchIntervalConformalSet, y_true: T) -> float:
    return empirical_coverage_regression(y_pred.tensor.cpu(), y_true)


@empirical_coverage_classification.register(TorchOneHotConformalSet)
def _empirical_coverage_classification_torch_onehot[T](y_pred: TorchOneHotConformalSet, y_true: T) -> float:
    return empirical_coverage_classification(y_pred.tensor.cpu(), y_true)
