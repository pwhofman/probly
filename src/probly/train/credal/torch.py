"""Torch losses for credal-set methods."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from probly.utils.torch import intersection_probability


def cvar_ce_loss(output: Tensor, targets: Tensor, delta: float) -> Tensor:
    """Cross-entropy averaged over the top ``floor(delta * B)`` highest-loss samples.

    The batch-wise CVaR approximation of Eq. 7 in
    :cite:`wangLearningCredalEnsembles2026`: only the worst ``delta`` fraction of the
    batch receives gradient. ``delta=1`` recovers the batch mean (ERM).

    Args:
        output: Logits of shape ``(B, num_classes)``.
        targets: Ground-truth class indices of shape ``(B,)``.
        delta: Fraction of highest-loss samples to keep, in (0, 1].

    Returns:
        Scalar cross-entropy loss averaged over the selected samples.

    Raises:
        ValueError: If delta is outside (0, 1].
    """
    if not 0.0 < delta <= 1.0:
        msg = f"delta must be in (0, 1], got {delta}."
        raise ValueError(msg)
    per_sample = F.cross_entropy(output, targets, reduction="none")
    if delta >= 1.0:
        return per_sample.mean()
    # floor(delta * B), clamped to 1 so degenerate tiny batches still train.
    k = max(1, int(delta * per_sample.shape[0]))
    return per_sample.topk(k).values.mean()


def intersection_probability_ce_loss(output: Tensor, targets: Tensor) -> Tensor:
    """Cross-entropy on the intersection probability of an interval-valued prediction.

    Implements Eq. 14 of :cite:`wangCredalDeepEnsembles2024`. Splits the packed
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
