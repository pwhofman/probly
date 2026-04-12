"""Torch implementations of conformal prediction metrics."""

from __future__ import annotations

import torch

from probly.representation.conformal_set.torch import TorchIntervalConformalSet, TorchOneHotConformalSet

from ._common import (
    average_interval_size,
    average_set_size,
    empirical_coverage_classification,
    empirical_coverage_regression,
)


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
