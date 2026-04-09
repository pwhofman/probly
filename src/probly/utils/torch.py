"""Utility functions for PyTorch models."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from .model_inspection import LAST_OUTPUT_DIM, output_dim_traverser


@output_dim_traverser.register(
    nn.Linear,
    vars={"_current": LAST_OUTPUT_DIM},
    update_vars=True,
)
def _(obj: nn.Linear, _current: int) -> tuple[nn.Linear, dict[str, int]]:
    """Record the output feature count of a torch ``nn.Linear`` layer."""
    return obj, {"_current": obj.out_features}


@output_dim_traverser.register(
    (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d),
    vars={"_current": LAST_OUTPUT_DIM},
    update_vars=True,
)
def _(obj: nn.Module, _current: int) -> tuple[nn.Module, dict[str, int]]:
    """Record the output channel count of a torch convolutional layer."""
    return obj, {"_current": obj.out_channels}  # ty: ignore[unresolved-attribute]


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
