"""Torch implementation for Kullback-Leibler divergence scores."""

from __future__ import annotations

import torch

from ._common import kl_divergence_score_func


@kl_divergence_score_func.register(torch.Tensor)
def compute_kl_divergence_score_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Computes the Kullback-Leibler divergence score using Torch Tensor."""
    y_pred_t = torch.asarray(y_pred)
    y_true_t = torch.asarray(y_true)

    if y_true_t.ndim == 1 or (y_true_t.shape[0] == 1 and y_true_t.numel() == y_pred_t.shape[0]):
        y_one_hot = torch.zeros_like(y_pred_t)
        y_one_hot[torch.arange(len(y_true_t)), y_true_t.flatten().int()] = 1.0
        y_true_t = y_one_hot

    eps = 1e-12
    y_pred_safe = torch.clamp(y_pred_t, min=eps)
    y_true_safe = torch.clamp(y_true_t, min=eps)

    return torch.sum(y_true_t * torch.log(y_true_safe / y_pred_safe), dim=-1)
