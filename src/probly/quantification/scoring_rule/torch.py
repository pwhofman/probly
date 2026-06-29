"""PyTorch implementations of scoring rule loss vectors."""

from __future__ import annotations

import torch

from ._common import _brier_loss_vector, _log_loss_vector, _spherical_loss_vector, _zero_one_loss_vector


@_log_loss_vector.register
def torch_log_loss_vector(probabilities: torch.Tensor) -> torch.Tensor:
    """Compute the per-label log loss vector for a torch tensor."""
    return -torch.log(probabilities)


@_brier_loss_vector.register
def torch_brier_loss_vector(probabilities: torch.Tensor) -> torch.Tensor:
    """Compute the per-label Brier loss vector for a torch tensor."""
    squared_norm = torch.sum(probabilities**2, dim=-1, keepdim=True)
    return squared_norm - 2.0 * probabilities + 1.0


@_zero_one_loss_vector.register
def torch_zero_one_loss_vector(probabilities: torch.Tensor) -> torch.Tensor:
    """Compute the per-label zero-one loss vector for a torch tensor."""
    num_classes = probabilities.shape[-1]
    argmax = torch.argmax(probabilities, dim=-1)
    one_hot = torch.nn.functional.one_hot(argmax, num_classes).to(probabilities.dtype)
    return 1.0 - one_hot


@_spherical_loss_vector.register
def torch_spherical_loss_vector(probabilities: torch.Tensor) -> torch.Tensor:
    """Compute the per-label spherical loss vector for a torch tensor."""
    norm = torch.sqrt(torch.sum(probabilities**2, dim=-1, keepdim=True))
    return 1.0 - probabilities / norm
