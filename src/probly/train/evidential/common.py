"""Unified Evidential Train Function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

import probly.train.evidential.torch as e
from probly.train.evidential.torch import der_loss, rpn_ng_kl, rpn_prior

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def unified_evidential_trainn(
    mode: str,
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
            if mode == "PostNet":
                loss, _ = loss_fn(outputs, y, flow, class_count)
            elif mode == "NatPostNet":
                alpha, _, _ = outputs
                loss = loss_fn(alpha, y)
            elif mode == "EDL":
                loss = loss_fn(outputs, y)  # calculate the evidential loss
            elif mode == "PrNet":
                total_loss = train_pn(model, optimizer, dataloader, oodloader)
                break
            elif mode == "IRD":
                x_adv = x + 0.01 * torch.randn_like(x)  # oder FGSM
                alpha = outputs
                alpha_adv = model(x_adv)
                y_oh = nn.functional.one_hot(y, num_classes=outputs.shape[1]).float()
                loss = loss_fn(alpha, y_oh, adversarial_alpha=alpha_adv)
            else:
                msg = "Enter valid mode"
                raise ValueError(msg)

            loss.backward()  # backpropagation
            optimizer.step()  # updates model-parameters

            total_loss += loss.item()  # add-up the loss of this epoch ontop of our total loss till then

        avg_loss = total_loss / len(dataloader)  # calculate average loss per epoch across all batches
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}")  # noqa: T201


def train_pn(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    id_loader: DataLoader,
    ood_loader: DataLoader,
) -> float:
    """Train the model for one epoch, using paired ID and OOD mini-batches."""
    device = "cpu"
    model.train()

    total_loss = 0.0

    ood_iter = iter(ood_loader)

    model.train()  # call of train important for models like dropout

    for x_in_raw, y_in_raw in id_loader:
        try:
            x_ood_raw, _ = next(ood_iter)
        except StopIteration:
            ood_iter = iter(ood_loader)
            x_ood_raw, _ = next(ood_iter)

        x_in = x_in_raw.to(device)
        y_in = y_in_raw.to(device)
        x_ood = x_ood_raw.to(device)

        optimizer.zero_grad()

        # In-distribution forward pass
        alpha_in = model(x_in)
        alpha_target_in = e.make_in_domain_target_alpha(y_in)
        kl_in = e.kl_dirichlet(alpha_target_in, alpha_in).mean()

        # Optional cross-entropy for classification stability
        probs_in = e.predictive_probs(alpha_in)
        ce_term = F.nll_loss(torch.log(probs_in + 1e-8), y_in)

        # OOD forward pass
        alpha_ood = model(x_ood)
        alpha_target_ood = e.make_ood_target_alpha(x_ood.size(0))
        kl_ood = e.kl_dirichlet(alpha_target_ood, alpha_ood).mean()

        # Total loss
        loss = kl_in + kl_ood + 0.1 * ce_term
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


def unified_loss(
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
