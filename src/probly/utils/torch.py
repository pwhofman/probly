"""Utility functions for PyTorch models."""

from __future__ import annotations

import torch
from tqdm import tqdm


@torch.no_grad()
def torch_collect_outputs(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect outputs and targets from a model for a given data loader.

    Args:
        model: torch.nn.Module, model to collect outputs from
        loader: torch.utils.data.DataLoader, data loader to collect outputs from
        device: torch.device, device to move data to
    Returns:
        outputs: torch.Tensor, shape (n_instances, n_classes), model outputs
        targets: torch.Tensor, shape (n_instances,), target labels
    """
    outputs = torch.empty(0, device=device)
    targets = torch.empty(0, device=device)
    for inpt, target in tqdm(loader):
        outputs = torch.cat((outputs, model(inpt.to(device))), dim=0)
        targets = torch.cat((targets, target.to(device)), dim=0)
    return outputs, targets


def torch_reset_all_parameters(module: torch.nn.Module) -> None:
    """Reset all parameters of a torch module.

    Args:
        module: torch.nn.Module, module to reset parameters

    """
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    for child in module.children():
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()
