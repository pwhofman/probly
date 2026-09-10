"""Torch implementation for Inner Product scores."""

from __future__ import annotations

import torch

from probly.representation.distribution.torch_categorical import TorchCategoricalDistribution
from probly.representation.sample.torch import TorchSample

from ._common import inner_product_score_func


@inner_product_score_func.register(torch.Tensor)
def compute_inner_product_score_torch(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Computes the Inner Product score using Torch Tensor."""
    y_pred_t = torch.asarray(y_pred)
    y_true_t = torch.asarray(y_true, device=y_pred_t.device)

    if y_true_t.shape == y_pred_t.shape[:-1] or (
        y_pred_t.ndim == 2 and y_true_t.shape == (1, y_pred_t.shape[0]) and y_true_t.shape != y_pred_t.shape
    ):
        labels = y_true_t.reshape(y_pred_t.shape[:-1]).long()
        return 1.0 - torch.gather(y_pred_t, -1, labels.unsqueeze(-1)).squeeze(-1)

    return 1.0 - torch.sum(y_pred_t * y_true_t, dim=-1)


@inner_product_score_func.register(TorchSample)
def _(y_pred: TorchSample, y_true: torch.Tensor) -> torch.Tensor:
    """Compute memberwise scores for Torch samples."""
    return inner_product_score_func(y_pred.tensor, y_true)


@inner_product_score_func.register(TorchCategoricalDistribution)
def _(y_pred: TorchCategoricalDistribution, y_true: torch.Tensor) -> torch.Tensor:
    """Compute the score from normalized categorical probabilities."""
    return inner_product_score_func(y_pred.probabilities, y_true)
