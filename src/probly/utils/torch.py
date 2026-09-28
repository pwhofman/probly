"""Utility functions for PyTorch models."""

from __future__ import annotations

from operator import index
from typing import Literal, SupportsIndex

import torch
import torch.nn.functional as F

from ._common import entropy, intersection_probability


def torch_head_dimension(head: torch.nn.Module, name: Literal["in_features", "out_features"]) -> int:
    """Read an integer feature dimension from a registered classification head.

    Custom heads may use any module class with the requested attribute. Integer-like
    values implementing ``__index__`` are supported, including NumPy integers.

    Args:
        head: Classification head selected by a traversal registration.
        name: Feature dimension needed by the consumer.

    Returns:
        The requested feature dimension.

    Raises:
        TypeError: If the head does not expose the requested integer dimension.
    """
    value = getattr(head, name, None)
    if not isinstance(value, SupportsIndex):
        msg = f"Classification head must expose an integer {name} attribute."
        raise TypeError(msg)
    return index(value)


def torch_temperature_softmax(logits: torch.Tensor, temperature: float | torch.Tensor) -> torch.Tensor:
    """Compute the softmax of logits with temperature scaling applied.

    Computes the softmax based on the logits divided by the temperature. Assumes that the last dimension
    of logits is the class dimension.

    Args:
        logits: Logits to apply softmax on of shape (n_instances, n_classes).
        temperature: Temperature scaling factor.

    Returns:
        Softmax of logits with temperature scaling applied of shape (n_instances, n_classes).

    """
    ts = F.softmax(logits / temperature, dim=-1)
    return ts


@entropy.register(torch.Tensor)
def torch_entropy(p: torch.Tensor) -> torch.Tensor:
    """Shannon entropy H(p) computed in torch along the last dim; 0*log(0) treated as 0.

    The logarithm is fed with the zeros replaced by ones instead of masking its result, so the
    gradient stays finite for probability vectors that contain exact zeros.

    Args:
        p: Probabilities to compute entropy of.

    Returns:
        Entropy of probabilities p
    """
    safe_p = torch.where(p > 0, p, torch.ones_like(p))
    result = -(p * safe_p.log()).sum(-1)
    return torch.clamp_min(result, 0.0) + 0.0  # Ensure non-negativity


@intersection_probability.register(torch.Tensor)
def torch_intersection_probability(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Intersection probability of a probability interval, per :cite:`wangCredalDeepEnsembles2024` Section 3.4.

    Reduces an interval credal set ``[lower, upper]`` to a single probability
    vector by ``q_int_k = lower_k + alpha * (upper_k - lower_k)`` with
    ``alpha = (1 - sum(lower)) / sum(upper - lower)``. The implementation
    handles the degenerate case ``upper == lower`` (zero width) by returning
    ``lower`` directly, avoiding ``0 / 0`` and keeping autograd well-defined.

    Args:
        lower: Lower bounds of shape ``(..., num_classes)``.
        upper: Upper bounds of shape ``(..., num_classes)``.

    Returns:
        Intersection probability tensor of shape ``(..., num_classes)``.
    """
    slack = upper - lower
    slack_sum = torch.sum(slack, dim=-1, keepdim=True)
    remaining = 1 - torch.sum(lower, dim=-1, keepdim=True)
    # Clamp alpha to [0, 1] to guard against floating-point drift where
    # sum(lower) slightly exceeds 1, which would make remaining negative and
    # produce sub-zero output probabilities.
    remaining = remaining.clamp(min=0)
    denominator = torch.where(slack_sum != 0, slack_sum, torch.ones_like(slack_sum))
    weights = torch.where(slack_sum != 0, slack / denominator, torch.zeros_like(slack))
    return lower + remaining * weights
