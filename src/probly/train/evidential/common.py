"""Unified Evidential Train Function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from probly.train.evidential.torch import PostNetLoss

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def unified_evidential_trainn(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.Tensor,
    flow: torch.Tensor = None,
    class_count: torch.Tensor = None,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cpu",
) -> None:
    """Demonstration of a unified evidential training function."""
    model = model.to(device)  # moves the model to the correct device (GPU or CPU)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # repeats the training function for a defined number of epochs
    for epoch in range(epochs):
        model.train()  # call of train important for models like dropout
        total_loss = 0.0  # track total_loss to calculate average loss per epoch

        for x, y in dataloader:
            # handle both cases: distributions (CIFAR10H original) or integer labels (fallback)
            x = x.to(device)  # noqa: PLW2901
            y = y.to(device)  # noqa: PLW2901

            optimizer.zero_grad()  # clears old gradients
            outputs = model(x)  # computes model-outputs
            if isinstance(loss_fn, PostNetLoss):
                loss, _ = loss_fn(outputs, y, flow, class_count)
            else:
                loss = loss_fn(outputs, y)  # calculate the evidential loss
            loss.backward()  # backpropagation
            optimizer.step()  # updates model-parameters

            total_loss += loss.item()  # add-up the loss of this epoch ontop of our total loss till then

        avg_loss = total_loss / len(dataloader)  # calculate average loss per epoch across all batches
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}")  # noqa: T201
