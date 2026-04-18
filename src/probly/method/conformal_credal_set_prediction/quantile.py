"""Calculate quantile."""

from __future__ import annotations

import numpy as np
import torch

from lazy_dispatch import lazydispatch


@lazydispatch
def calculate_quantile[In](scores: In, alpha: float) -> In:
    """Calculate the conformal quantile from nonconformity scores."""
    msg = "Quantile score computation not implemented for this type."
    raise NotImplementedError(msg)


@calculate_quantile.register(np.ndarray)
def calculate_quantile_numpy(scores: np.ndarray, alpha: float) -> float:
    """Calculate the quantile for conformal prediction."""
    n = len(scores)
    alpha_prime = np.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(scores, alpha_prime, method="higher"))


@calculate_quantile.register(torch.Tensor)
def _compute_quantile_score_torch(scores: torch.Tensor, alpha: float) -> float:
    """Calculate the quantile für conformal prediction using tensor."""
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

        return float(torch.quantile(scores, q_level, interpolation="nearest"))
