"""Unified Evidential Regression (DER + RPN)."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class EvidentialRegression(nn.Module):
    """Evidential regression model.

    Outputs parameters of a 1D Normal-Inverse-Gamma / Normal-Gamma:
    mu, kappa, alpha, beta.
    """

    def __init__(self) -> None:
        """Initialize MLP architecture for evidential regression."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return mu, kappa, alpha, beta parameters for the predictive NIG."""
        out = self.layers(x)

        mu = out[:, 0:1]
        kappa = F.softplus(out[:, 1:2])  # ≥ 0
        alpha = F.softplus(out[:, 2:3]) + 1.0  # > 1
        beta = F.softplus(out[:, 3:4])  # > 0

        return mu, kappa, alpha, beta


def der_loss(
    y: Tensor,
    mu: Tensor,
    kappa: Tensor,
    alpha: Tensor,
    beta: Tensor,
    lam: float = 0.01,
) -> Tensor:
    """Deep Evidential Regression loss (Student-t NLL + evidence regularizer)."""
    eps = 1e-8
    two_bv = 2.0 * beta * (1.0 + kappa) + eps

    lnll = (
        0.5 * torch.log(torch.pi / (kappa + eps))
        - alpha * torch.log(two_bv)
        + (alpha + 0.5) * torch.log(kappa * (y - mu) ** 2 + two_bv)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    evidence = 2.0 * kappa + alpha
    reg = torch.abs(y - mu) * evidence

    return (lnll + lam * reg).mean()


def rpn_prior(
    shape: torch.Size | tuple[int, ...],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return zero-evidence Normal-Gamma prior parameters for RPN KL."""
    eps = 1e-6
    mu0 = torch.zeros(shape, device=device)
    kappa0 = torch.ones(shape, device=device) * eps
    alpha0 = torch.ones(shape, device=device) * (1.0 + eps)
    beta0 = torch.ones(shape, device=device) * eps
    return mu0, kappa0, alpha0, beta0


def rpn_ng_kl(
    mu: Tensor,
    kappa: Tensor,
    alpha: Tensor,
    beta: Tensor,
    mu0: Tensor,
    kappa0: Tensor,
    alpha0: Tensor,
    beta0: Tensor,
) -> Tensor:
    """Compute KL divergence between two 1D Normal-Gamma distributions."""
    eps = 1e-8

    kappa = kappa + eps
    kappa0 = kappa0 + eps
    beta = beta + eps
    beta0 = beta0 + eps

    ratio_kappa = kappa / kappa0
    term_mu = 0.5 * (alpha / beta) * kappa0 * (mu - mu0).pow(2)
    term_kappa = 0.5 * (ratio_kappa - torch.log(ratio_kappa) - 1.0)
    term_gamma = (
        alpha0 * torch.log(beta / beta0)
        - torch.lgamma(alpha)
        + torch.lgamma(alpha0)
        + (alpha - alpha0) * torch.digamma(alpha)
        - (beta - beta0) * (alpha / beta)
    )

    return (term_mu + term_kappa + term_gamma).mean()


def unified_loss(
    y: Tensor,
    mu: Tensor,
    kappa: Tensor,
    alpha: Tensor,
    beta: Tensor,
    is_ood: Tensor,
    lam_der: float = 0.01,
    lam_rpn: float = 1.0,
) -> Tensor:
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
