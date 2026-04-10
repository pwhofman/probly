"""PyTorch implementation of ROC curve."""

from __future__ import annotations

import torch

from probly.metrics import roc_curve


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
