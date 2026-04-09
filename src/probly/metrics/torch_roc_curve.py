"""PyTorch implementation of ROC curve."""

from __future__ import annotations

import torch

from probly.metrics import roc_curve


@roc_curve.register(torch.Tensor)
def roc_curve_torch(y_true: torch.Tensor, y_score: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ROC curve for PyTorch tensors."""
    y_true = y_true.float()
    y_score = y_score.float()
    if y_true.ndim == 2:
        return _roc_curve_torch_batched(y_true, y_score)

    desc_idx = torch.argsort(y_score, stable=True, descending=True)
    y_score_sorted = y_score[desc_idx]
    y_true_sorted = y_true[desc_idx]

    distinct_mask = torch.diff(y_score_sorted) != 0
    distinct_idx = torch.where(distinct_mask)[0]
    threshold_idx = torch.cat([distinct_idx, torch.tensor([len(y_true) - 1], device=y_true.device)])

    tps = torch.cumsum(y_true_sorted, dim=0)[threshold_idx]
    fps = (threshold_idx + 1).float() - tps

    total_pos = y_true.sum()
    total_neg = float(len(y_true)) - total_pos

    tpr = tps / total_pos if total_pos > 0 else torch.zeros_like(tps)
    fpr = fps / total_neg if total_neg > 0 else torch.zeros_like(fps)

    zero = torch.zeros(1, device=y_true.device)
    tpr = torch.cat([zero, tpr])
    fpr = torch.cat([zero, fpr])
    thresholds = torch.cat([y_score_sorted[:1] + 1, y_score_sorted[threshold_idx]])

    return fpr, tpr, thresholds


def _roc_curve_torch_batched(
    y_true: torch.Tensor, y_score: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = y_true.shape[1]

    desc_idx = torch.argsort(y_score, dim=-1, stable=True, descending=True)
    y_score_sorted = torch.gather(y_score, -1, desc_idx)
    y_true_sorted = torch.gather(y_true, -1, desc_idx)

    tps = torch.cumsum(y_true_sorted, dim=-1)
    fps = torch.arange(1, n + 1, device=y_true.device, dtype=y_true.dtype).unsqueeze(0) - tps

    total_pos = y_true.sum(dim=-1, keepdim=True)
    total_neg = n - total_pos

    tpr = torch.where(total_pos > 0, tps / total_pos, torch.zeros_like(tps))
    fpr = torch.where(total_neg > 0, fps / total_neg, torch.zeros_like(fps))

    zeros = torch.zeros(y_true.shape[0], 1, device=y_true.device)
    tpr = torch.cat([zeros, tpr], dim=-1)
    fpr = torch.cat([zeros, fpr], dim=-1)
    thresholds = torch.cat([y_score_sorted[:, :1] + 1, y_score_sorted], dim=-1)

    return fpr, tpr, thresholds
