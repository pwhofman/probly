"""Torch losses for credal-set methods."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from probly.utils.torch import intersection_probability


def intersection_probability_ce_loss(output: Tensor, targets: Tensor) -> Tensor:
    """Cross-entropy on the intersection probability of an interval-valued prediction.

    Implements Eq. 14 of :cite:`wang2024credalnet`. Splits the packed
    ``(B, 2C)`` interval output into ``(lower, upper)``, computes the
    intersection probability, and applies negative-log-likelihood against
    the targets. The probabilities are clamped to ``finfo(dtype).eps``
    before the log to avoid ``-inf``.

    Args:
        output: Packed ``(B, 2 * num_classes)`` tensor with the lower bounds
            in the first half and the upper bounds in the second.
        targets: Ground-truth class indices of shape ``(B,)``.

    Returns:
        Scalar cross-entropy loss averaged over the batch.
    """
    n_classes = output.shape[-1] // 2
    q_int = intersection_probability(output[..., :n_classes], output[..., n_classes:])
    eps = torch.finfo(q_int.dtype).eps
    return F.nll_loss(torch.log(q_int.clamp(min=eps)), targets)
