"""Torch implementations of active learning metrics."""

from __future__ import annotations

import torch

from probly.train.calibration.torch import ExpectedCalibrationError

from .metrics import compute_accuracy, compute_ece


@compute_accuracy.register(torch.Tensor)
def _torch_compute_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return float((y_pred == y_true).float().mean().item())


@compute_ece.register(torch.Tensor)
def _torch_compute_ece(probs: torch.Tensor, y_true: torch.Tensor, n_bins: int = 10) -> float:
    ece_fn = ExpectedCalibrationError(num_bins=n_bins)
    with torch.no_grad():
        loss = ece_fn(probs.float(), y_true.long())
    return float(loss.item())
