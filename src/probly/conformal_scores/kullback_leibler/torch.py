"""Torch implementation for Kullback-Leibler divergence scores."""

from __future__ import annotations

import torch

from ._common import kl_divergence_score_func


@kl_divergence_score_func.register(torch.Tensor)
def compute_kl_divergence_score_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Computes the Kullback-Leibler divergence score using Torch Tensor."""
    y_pred_t = torch.asarray(y_pred)
    y_true_t = torch.asarray(y_true, device=y_pred_t.device)

    eps = 1e-12
    if y_true_t.shape == y_pred_t.shape[:-1] or (
        y_pred_t.ndim == 2 and y_true_t.shape == (1, y_pred_t.shape[0]) and y_true_t.shape != y_pred_t.shape
    ):
        labels = y_true_t.reshape(y_pred_t.shape[:-1]).long()
        probabilities = torch.gather(y_pred_t, -1, labels.unsqueeze(-1)).squeeze(-1)
        return -torch.log(torch.clamp(probabilities, min=eps))

    y_pred_safe = torch.clamp(y_pred_t, min=eps)
    y_true_safe = torch.clamp(y_true_t, min=eps)

    return torch.sum(y_true_t * torch.log(y_true_safe / y_pred_safe), dim=-1)
