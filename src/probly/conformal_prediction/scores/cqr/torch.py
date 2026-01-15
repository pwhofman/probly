"""PyTorch implementation for CQR scores."""

from __future__ import annotations

import torch
from torch import Tensor

from probly.conformal_prediction.scores.cqr.common import register


def cqr_score_torch(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """PyTorch implementation of CQR nonconformity score.

    Computes: s = max(q_lo - y, y - q_hi, 0)

    This implementation preserves gradients for backpropagation.
    """
    # Ensure y_true is flat (N,)
    y = y_true.reshape(-1)

    # y_pred must be (N, 2)
    if y_pred.ndim != 2 or y_pred.shape[1] != 2:
        msg = f"y_pred must have shape (n_samples, 2), got {y_pred.shape}"
        raise ValueError(msg)

    # Extract lower and upper quantiles
    lower = y_pred[:, 0]
    upper = y_pred[:, 1]

    # Calculate differences (preserving gradients)
    diff_lower = lower - y
    diff_upper = y - upper

    # Element-wise maximum of the two differences
    # equivalent to np.maximum(diff_lower, diff_upper)
    max_diff = torch.maximum(diff_lower, diff_upper)

    # Apply ReLU logic: max(value, 0)
    # torch.clamp is differentiable
    scores = torch.clamp(max_diff, min=0.0)

    return scores


# Register implementation for PyTorch Tensors
# FIX: Register directly with the class 'torch.Tensor' instead of LazyType.
# This avoids 'TypeAliasType not callable' errors in Python 3.12+.
register(torch.Tensor, cqr_score_torch)
