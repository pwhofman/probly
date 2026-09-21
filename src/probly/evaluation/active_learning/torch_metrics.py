"""Torch implementations of active learning metrics."""

from __future__ import annotations

import torch

from probly.metrics.torch import torch_expected_calibration_error

from .metrics import compute_accuracy, compute_ece


@compute_accuracy.register(torch.Tensor)
def _torch_compute_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return float((y_pred == y_true).float().mean().item())


@compute_ece.register(torch.Tensor)
def _torch_compute_ece(probs: torch.Tensor, y_true: torch.Tensor, n_bins: int = 10) -> float:
    with torch.no_grad():
        loss = torch_expected_calibration_error(probs.float(), y_true.long(), num_bins=n_bins)
    return float(loss.item())
