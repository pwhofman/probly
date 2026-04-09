"""PyTorch implementation of ROC AUC score."""

from __future__ import annotations

import torch

from probly.metrics import auc, roc_auc_score, roc_curve


@roc_auc_score.register(torch.Tensor)
def roc_auc_score_torch(y_true: torch.Tensor, y_score: torch.Tensor) -> torch.Tensor:
    """Compute area under the ROC curve for PyTorch tensors."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)
