"""Utility functions for PyTorch models."""

from __future__ import annotations

from operator import index
from typing import Literal, SupportsIndex

import torch
import torch.nn.functional as F
from tqdm import tqdm


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


@torch.no_grad()
def torch_collect_outputs(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect outputs and targets from a model for a given data loader.

    Args:
        model: Model to collect outputs from.
        loader: Data loader to collect outputs from.
        device: Device to move data to.

    Returns:
        A tuple containing:
            - outputs: Model outputs of shape (n_instances, n_classes).
            - targets: Target labels of shape (n_instances,).

    """
    outputs = torch.empty(0, device=device)
    targets = torch.empty(0, device=device)
    for inpt, target in tqdm(loader, desc="Batches"):
        outputs = torch.cat((outputs, model(inpt.to(device))), dim=0)
        targets = torch.cat((targets, target.to(device)), dim=0)
    return outputs, targets


def torch_reset_all_parameters(module: torch.nn.Module) -> None:
    """Reset all parameters of a torch module.

    Args:
        module: Module to reset parameters.

    """
    reset = getattr(module, "reset_parameters", None)
    if callable(reset):
        reset()
    for child in module.children():
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()


def temperature_softmax(logits: torch.Tensor, temperature: float | torch.Tensor) -> torch.Tensor:
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


def torch_entropy(p: torch.Tensor) -> torch.Tensor:
    """Shannon entropy H(p) computed in torch along the last dim; 0*log(0) treated as 0.

    Args:
        p: Probabilities to compute entropy of.

    Returns:
        Entropy of probabilities p
    """
    log_p = torch.where(p > 0, p.log(), p.new_zeros(()))
    result = -(p * log_p).sum(-1)
    return torch.clamp_min(result, 0.0) + 0.0  # Ensure non-negativity


def intersection_probability(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
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


def dirichlet_entropy(alphas: torch.Tensor) -> torch.Tensor:
    """Compute the differential entropy of Dirichlet distributions with concentrations of shape ``(..., K)``."""
    alpha_0 = torch.sum(alphas, dim=-1)
    num_classes = alphas.shape[-1]

    log_beta = torch.sum(torch.lgamma(alphas), dim=-1) - torch.lgamma(alpha_0)
    digamma_sum = (alpha_0 - num_classes) * torch.digamma(alpha_0)
    digamma_individual = torch.sum((alphas - 1) * torch.digamma(alphas), dim=-1)
    return log_beta + digamma_sum - digamma_individual
