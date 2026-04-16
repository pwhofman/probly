"""Torch implementation for CQR scores."""

from __future__ import annotations

import torch

from probly.representation.sample.torch import TorchSample

from ._common import cqr_score_func


@cqr_score_func.register(torch.Tensor)
def compute_cqr_score_func_torch(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """CQR nonconformity scores for PyTorch tensors."""
    y = y_true.reshape(-1)

    if y_pred.ndim > 3 or y_pred.shape[-1] != 2:
        msg = (
            "y_pred must have shape (n_evaluations, n_samples, 2), "
            f"got {y_pred.shape}. The n_evaluations dimension is optional and "
            "will be averaged over if present."
        )
        raise ValueError(msg)
    if y_pred.ndim == 3:
        y_pred = y_pred.mean(dim=0)

    lower = y_pred[:, 0]
    upper = y_pred[:, 1]

    return torch.maximum(lower - y, y - upper)


@cqr_score_func.register(TorchSample)
def _(y_pred: TorchSample, y_true: torch.Tensor) -> torch.Tensor:
    """CQR nonconformity scores for TorchSamples."""
    return compute_cqr_score_func_torch(y_pred.tensor, y_true)
