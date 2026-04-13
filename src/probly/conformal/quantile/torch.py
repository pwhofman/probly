"""PyTorch implementations of conformal quantile computation."""

from __future__ import annotations

import numpy as np
import torch

from ._common import calculate_quantile, calculate_weighted_quantile


@calculate_quantile.register(torch.Tensor)
def _compute_quantile_score_torch(scores: torch.Tensor, alpha: float) -> float:
    # Implementation for PyTorch tensors
    with torch.no_grad():
        if not 0 <= alpha <= 1:
            msg = f"alpha must be in [0, 1], got {alpha}"
            raise ValueError(msg)

        n = len(scores)
        if n == 0:
            msg = "scores array is empty"
            raise ValueError(msg)

        q_level = torch.ceil(torch.tensor((n + 1) * (1 - alpha))) / n
        q_level = torch.minimum(q_level, torch.tensor(1.0))  # ensure within [0, 1]

        # Inverted CDF / right-continuous step quantile
        # PyTorch does not support "inverted_cdf" method; "nearest" is the most precise available approximation.
        return float(torch.quantile(scores, q_level, interpolation="nearest"))


@calculate_weighted_quantile.register(torch.Tensor)
def _compute_weighted_quantile_torch(
    values: torch.Tensor, quantile: float, sample_weight: torch.Tensor | None = None
) -> float:
    with torch.no_grad():
        if sample_weight is None:
            return float(torch.quantile(values, quantile, interpolation="linear"))

        values = torch.tensor(values)
        sample_weight = torch.tensor(sample_weight)

        sorter = torch.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

        weighted_quantiles = torch.cumsum(sample_weight, dim=0) - 0.5 * sample_weight
        weighted_quantiles /= torch.sum(sample_weight)

        return float(np.interp(quantile, weighted_quantiles.numpy(), values.numpy()))
