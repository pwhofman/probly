"""Unified Evidential Train Function."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from probly.train.evidential.torch import der_loss, rpn_ng_kl, rpn_prior, train_pn

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def unified_evidential_train_class(  # noqa: C901, PLR0912
    mode: Literal["PostNet", "NatPostNet", "EDL", "PrNet", "IRD"],
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.Tensor = None,
    oodloader: DataLoader = None,
    flow: torch.Tensor = None,
    class_count: torch.Tensor = None,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cpu",
) -> None:
    """Trains a given Neural Network using different learning approaches, depending on the approach of a selected paper.

    Args:
        mode:
            Identifier of the paper-based training approach to be used.
            Must be one of:
            "PostNet", "NatPostNet", "EDL", "PrNet", or "IRD".

        model:
            The neural network to be trained.

        dataloader:
            Pytorch.Dataloader providing the In-Distributtion training samples and corresponding labels.

        loss_fn:
            Loss functions used for training. The inputs of each loss-functions depends on the selected mode

        oodloader:
            Pytorch.Dataloader providing the Out-Of-Distributtion training samples and corresponding labels.
            This is only required for certain modes such as "PrNet"

        flow:
            Optional normalizing flow module used by posterior network-based methods like "PostNet"

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
    if flow is not None:
        flow = flow.to(device)

    if mode == "PostNet":
        if flow is None:
            msg = "PostNet mode requires a flow module."
            raise ValueError(msg)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(flow.parameters()),
            lr=lr,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # repeats the training function for a defined number of epochs
    for epoch in range(epochs):
        model.train()  # call of train important for models like dropout
        if flow is not None and mode == "PostNet":
            flow.train()
        total_loss = 0.0  # track total_loss to calculate average loss per epoch

        for x, y in dataloader:
            x = x.to(device)  # noqa: PLW2901
            y = y.to(device)  # noqa: PLW2901

            optimizer.zero_grad()  # clears old gradients
            outputs = model(x)  # computes model-outputs
            if mode == "PostNet":
                loss, _ = loss_fn(outputs, y, flow, class_count)
            elif mode == "NatPostNet":
                alpha, _, _ = outputs
                loss = loss_fn(alpha, y)
            elif mode == "EDL":
                loss = loss_fn(outputs, y)
            elif mode == "PrNet":
                total_loss = train_pn(model, optimizer, dataloader, oodloader)
                break
            elif mode == "IRD":
                x_adv = x + 0.01 * torch.randn_like(x)
                alpha = outputs
                alpha_adv = model(x_adv)
                y_oh = nn.functional.one_hot(y, num_classes=outputs.shape[1]).float()
                loss = loss_fn(alpha, y_oh, adversarial_alpha=alpha_adv)
            else:
                msg = "Enter valid mode"
                raise ValueError(msg)

            loss.backward()  # backpropagation
            optimizer.step()  # updates model-parameters

            total_loss += loss.item()  # add-up the loss of this epoch ontop of our total loss

        avg_loss = total_loss / len(dataloader)  # calculate average loss per epoch across all batches
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}")  # noqa: T201


def unified_evidential_train_reg(
    y: torch.Tensor,
    mu: torch.Tensor,
    kappa: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    is_ood: torch.Tensor,
    lam_der: float = 0.01,
    lam_rpn: float = 1.0,
) -> torch.Tensor:
    """Compute unified DER + RPN loss.

    Deep Evidential Regression is applied to in-distribution samples, while a
    Normal-Gamma KL divergence (RPN) is applied to out-of-distribution samples.
    """
    is_ood_bool = is_ood.bool()
    id_mask = ~is_ood_bool
    ood_mask = is_ood_bool

    device = y.device
    loss_id = torch.tensor(0.0, device=device)
    loss_ood = torch.tensor(0.0, device=device)

    if id_mask.any():
        loss_id = der_loss(
            y[id_mask],
            mu[id_mask],
            kappa[id_mask],
            alpha[id_mask],
            beta[id_mask],
            lam=lam_der,
        )

    if ood_mask.any():
        shape = mu[ood_mask].shape
        mu0, kappa0, alpha0, beta0 = rpn_prior(shape, device)

        loss_ood = rpn_ng_kl(
            mu[ood_mask],
            kappa[ood_mask],
            alpha[ood_mask],
            beta[ood_mask],
            mu0,
            kappa0,
            alpha0,
            beta0,
        )

    return loss_id + lam_rpn * loss_ood
