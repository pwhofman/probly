# ruff: noqa

# ============================================================
#   Unified Evidential Regression (DER + RPN)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

class EvidentialRegression(nn.Module):
    """
    Model outputs parameters of a Normal-Inverse-Gamma (equivalent
    to Normal-Gamma in 1D) distribution:
        mu, kappa, alpha, beta
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        out = self.layers(x)

        mu    = out[:, 0:1]
        kappa = F.softplus(out[:, 1:2])         # ≥ 0
        alpha = F.softplus(out[:, 2:3]) + 1.0   # > 1
        beta  = F.softplus(out[:, 3:4])         # > 0

        return mu, kappa, alpha, beta


# ------------------------------------------------------------
# DER LOSS (CORRECT FROM PAPER)
# ------------------------------------------------------------

def der_loss(y, mu, kappa, alpha, beta, lam=0.01):
    """
    Deep Evidential Regression loss.
    """
    eps = 1e-8
    two_bv = 2.0 * beta * (1.0 + kappa) + eps

    # Student-t NLL
    lnll = (
        0.5 * torch.log(torch.pi / (kappa + eps))
        - alpha * torch.log(two_bv)
        + (alpha + 0.5) * torch.log(kappa * (y - mu) ** 2 + two_bv)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    # Evidence regularizer
    evidence = 2*kappa + alpha
    reg = torch.abs(y - mu) * evidence

    return (lnll + lam * reg).mean()


# ------------------------------------------------------------
# RPN PRIOR (ZERO-EVIDENCE NORMAL-GAMMA)
# ------------------------------------------------------------

def rpn_prior(shape, device):
    """
    Zero-evidence Normal-Gamma prior for RPN loss.
    """
    eps = 1e-6
    mu0    = torch.zeros(shape, device=device)
    kappa0 = torch.ones(shape, device=device) * eps
    alpha0 = torch.ones(shape, device=device) * (1 + eps)
    beta0  = torch.ones(shape, device=device) * eps
    return mu0, kappa0, alpha0, beta0


# ------------------------------------------------------------
# TRUE NORMAL-GAMMA KL (RPN LOSS)
# ------------------------------------------------------------

def rpn_ng_kl(mu, kappa, alpha, beta,
              mu0, kappa0, alpha0, beta0):
    """
    KL divergence between 1D Normal-Gamma distributions.
    This **IS** the correct RPN regression KL loss.
    """
    eps = 1e-8
    kappa  = kappa  + eps
    kappa0 = kappa0 + eps
    beta   = beta   + eps
    beta0  = beta0  + eps

    # Term 1: KL on the Normal means
    term_mu = 0.5 * (alpha / beta) * kappa0 * (mu - mu0)**2

    # Term 2: KL on precision mixture
    term_kappa = 0.5 * (kappa / kappa0 - torch.log(kappa / kappa0) - 1)

    # Term 3: KL on Gamma distributions
    term_gamma = (
        alpha0 * torch.log(beta / beta0)
        - torch.lgamma(alpha) + torch.lgamma(alpha0)
        + (alpha - alpha0) * torch.digamma(alpha)
        - (beta - beta0) * (alpha / beta)
    )

    return (term_mu + term_kappa + term_gamma).mean()


# ------------------------------------------------------------
# UNIFIED LOSS (DER for ID, RPN KL for OOD)
# ------------------------------------------------------------

def unified_loss(y, mu, kappa, alpha, beta, is_ood,
                 lam_der=0.01, lam_rpn=1.0):

    is_ood = is_ood.bool()
    id_mask  = ~is_ood
    ood_mask =  is_ood

    device = y.device

    loss_id  = torch.tensor(0.0, device=device)
    loss_ood = torch.tensor(0.0, device=device)

    # ------------------------
    # DER loss for ID samples
    # ------------------------
    if id_mask.any():
        loss_id = der_loss(
            y[id_mask],
            mu[id_mask],
            kappa[id_mask],
            alpha[id_mask],
            beta[id_mask],
            lam=lam_der
        )

    # ------------------------
    # RPN KL loss for OOD samples
    # ------------------------
    if ood_mask.any():
        shape = mu[ood_mask].shape
        mu0, kappa0, alpha0, beta0 = rpn_prior(shape, device)

        loss_ood = rpn_ng_kl(
            mu[ood_mask], kappa[ood_mask], alpha[ood_mask], beta[ood_mask],
            mu0,          kappa0,          alpha0,          beta0
        )

    return loss_id + lam_rpn * loss_ood
