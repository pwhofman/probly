"""PyTorch implementation of precision-recall curve."""

from __future__ import annotations

import torch

from probly.metrics import precision_recall_curve


@precision_recall_curve.register(torch.Tensor)
def precision_recall_curve_torch(
    y_true: torch.Tensor, y_score: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute precision-recall curve for PyTorch tensors."""
    y_true = y_true.float()
    y_score = y_score.float()
    if y_true.ndim == 2:
        return _precision_recall_curve_torch_batched(y_true, y_score)

    desc_idx = torch.argsort(y_score, stable=True, descending=True)
    y_score_sorted = y_score[desc_idx]
    y_true_sorted = y_true[desc_idx]

    distinct_mask = torch.diff(y_score_sorted) != 0
    distinct_idx = torch.where(distinct_mask)[0]
    threshold_idx = torch.cat([distinct_idx, torch.tensor([len(y_true) - 1], device=y_true.device)])

    tps = torch.cumsum(y_true_sorted, dim=0)[threshold_idx]
    predicted_pos = (threshold_idx + 1).float()
    total_pos = y_true.sum()

    precision = tps / predicted_pos
    recall = tps / total_pos if total_pos > 0 else torch.zeros_like(tps)

    precision = torch.cat([precision.flip(0), torch.ones(1, device=y_true.device)])
    recall = torch.cat([recall.flip(0), torch.zeros(1, device=y_true.device)])

    return precision, recall, y_score_sorted[threshold_idx]


def _precision_recall_curve_torch_batched(
    y_true: torch.Tensor, y_score: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = y_true.shape[1]

    desc_idx = torch.argsort(y_score, dim=-1, stable=True, descending=True)
    y_score_sorted = torch.gather(y_score, -1, desc_idx)
    y_true_sorted = torch.gather(y_true, -1, desc_idx)

    tps = torch.cumsum(y_true_sorted, dim=-1)
    predicted_pos = torch.arange(1, n + 1, device=y_true.device, dtype=y_true.dtype).unsqueeze(0)
    total_pos = y_true.sum(dim=-1, keepdim=True)

    precision = tps / predicted_pos
    recall = torch.where(total_pos > 0, tps / total_pos, torch.zeros_like(tps))

    precision = torch.cat([precision.flip(-1), torch.ones(y_true.shape[0], 1, device=y_true.device)], dim=-1)
    recall = torch.cat([recall.flip(-1), torch.zeros(y_true.shape[0], 1, device=y_true.device)], dim=-1)

    return precision, recall, y_score_sorted
