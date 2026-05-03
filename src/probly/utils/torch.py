"""Utility functions for PyTorch models."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm


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
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()  # ty: ignore[call-non-callable]
    for child in module.children():
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()  # ty: ignore[call-non-callable]


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
    """Intersection probability of a probability interval, per :cite:`wang2024credalnet` Section 3.4.

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
    denominator = torch.where(slack_sum != 0, slack_sum, torch.ones_like(slack_sum))
    weights = torch.where(slack_sum != 0, slack / denominator, torch.zeros_like(slack))
    return lower + remaining * weights
