"""Unified Evidential Train Function."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn

from probly.utils.switchdispatch import switchdispatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.data import DataLoader


def unified_evidential_train(
    mode: Literal["PostNet", "NatPostNet", "EDL", "PrNet", "IRD", "DER", "RPN"],
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable[..., torch.Tensor] | None = None,
    oodloader: DataLoader | None = None,
    class_count: torch.Tensor | None = None,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cpu",
) -> None:
    """Trains a given Neural Network using different learning approaches, depending on the approach of a selected paper.

    Args:
        mode:
            Identifier of the paper-based training approach to be used.
            Must be one of:
            "PostNet", "NatPostNet", "EDL", "PrNet", "IRD", "DER" or "RPN".

        model:
            The neural network to be trained.

        dataloader:
            Pytorch.Dataloader providing the In-Distributtion training samples and corresponding labels.

        loss_fn:
            Loss functions used for training. The inputs of each loss-functions depends on the selected mode

        oodloader:
            Pytorch.Dataloader providing the Out-Of-Distributtion training samples and corresponding labels.
            This is only required for certain modes such as "PrNet"

        class_count:
            Tensor containing the number of samples per class.

        epochs:
            Number of training epochs.

        lr:
            Learning rate used by the optimizer.

        device:
            Device on which the model is trained
            (e.g. "cpu" or "cuda")

    Returns:
        None.
        The function performs training of the provided model and does not return a value.
        But prints the total-losses per Epoch.
    """
    model = model.to(device)  # moves the model to the correct device (GPU or CPU)

    if mode == "PostNet" and not hasattr(model, "flow"):
        msg = "PostNet mode requires a flow module."
        raise ValueError(msg)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # repeats the training function for a defined number of epochs
    for epoch in range(epochs):
        model.train()  # call of train important for models like dropout
        total_loss = 0.0  # track total_loss to calculate average loss per epoch

        for x_raw, y_raw in dataloader:
            x = x_raw.to(device)
            y = y_raw.to(device)

            optimizer.zero_grad()  # clears old gradients
            outputs = model(x)  # computes model-outputs

            loss = compute_loss(
                mode,
                outputs=outputs,
                loss_fn=loss_fn,
                model=model,
                x=x,
                y=y,
                device=torch.device(device),
                oodloader=oodloader,
                class_count=class_count,
            )  # computes the loss based on the selected mode

            loss.backward()  # backpropagation
            optimizer.step()  # updates model-parameters

            total_loss += loss.item()  # add-up the loss of this epoch ontop of our total loss

        avg_loss = total_loss / len(dataloader)  # calculate average loss per epoch across all batches
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}")  # noqa: T201


@switchdispatch
def compute_loss(
    _mode: str,
    **_kwargs: dict[str, Any],
) -> torch.Tensor:
    """Dispatch function for computing the loss based on the selected mode via switchdispatch."""
    msg = 'Enter a valid mode ["PostNet", "NatPostNet", "EDL", "PrNet", "IRD", "DER", "RPN"]'
    raise ValueError(msg)


@compute_loss.register("PostNet")
def _postnet_loss(
    _mode: str,
    *,
    y: torch.Tensor,
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    loss_fn: Callable[..., torch.Tensor],
    **_kwargs: dict[str, Any],
) -> torch.Tensor:
    alpha, _, _ = outputs
    return loss_fn(alpha, y)


@compute_loss.register("NatPostNet")
def _natpostnet_loss(
    _mode: str,
    *,
    y: torch.Tensor,
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    loss_fn: Callable[..., torch.Tensor],
    **_kwargs: dict[str, Any],
) -> torch.Tensor:
    alpha, _, _ = outputs
    return loss_fn(alpha, y)


@compute_loss.register("EDL")
def _edl_loss(
    _mode: str,
    *,
    y: torch.Tensor,
    outputs: torch.Tensor,
    loss_fn: Callable[..., torch.Tensor],
    **_kwargs: dict[str, Any],
) -> torch.Tensor:
    return loss_fn(outputs, y)


@compute_loss.multi_register({"PrNet", "RPN"})
def _prnet_rpn_loss(
    _mode: str,
    *,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable[..., torch.Tensor],
    oodloader: DataLoader,
    **_: dict[str, Any],
) -> torch.Tensor:
    ood_iter = iter(oodloader)
    try:
        x_ood_raw, _ = next(ood_iter)
    except StopIteration:
        ood_iter = iter(oodloader)
        x_ood_raw, _ = next(ood_iter)

    x_ood = x_ood_raw.to(x.device)
    return loss_fn(model, x, y, x_ood)


@compute_loss.register("IRD")
def _ird_loss(
    _mode: str,
    *,
    y: torch.Tensor,
    outputs: torch.Tensor,
    loss_fn: Callable[..., torch.Tensor],
    model: nn.Module,
    x: torch.Tensor,
    **_kwargs: dict[str, Any],
) -> torch.Tensor:
    x_adv = x + 0.01 * torch.randn_like(x)
    alpha = outputs
    alpha_adv = model(x_adv)
    y_oh = nn.functional.one_hot(
        y,
        num_classes=outputs.shape[1],
    ).float()
    return loss_fn(alpha, y_oh, adversarial_alpha=alpha_adv)


@compute_loss.register("DER")
def _der_loss(
    _mode: str,
    *,
    y: torch.Tensor,
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    loss_fn: Callable[..., torch.Tensor],
    **_kwargs: dict[str, Any],
) -> torch.Tensor:
    mu, kappa, alpha, beta = outputs
    return loss_fn(y, mu, kappa, alpha, beta)
