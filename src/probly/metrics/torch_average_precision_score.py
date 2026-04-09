"""PyTorch implementations of classification metrics."""

from __future__ import annotations

import torch

from ._common import average_precision_score, precision_recall_curve


@average_precision_score.register(torch.Tensor)
def average_precision_score_torch(y_true: torch.Tensor, y_score: torch.Tensor) -> torch.Tensor:
    """Compute average precision for PyTorch tensors."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return -torch.sum(torch.diff(recall, dim=-1) * precision[..., :-1], dim=-1)
