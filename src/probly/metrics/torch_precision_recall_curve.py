"""PyTorch implementation of precision-recall curve."""

from __future__ import annotations

import torch

from probly.metrics import precision_recall_curve


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
