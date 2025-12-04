"""Collection of torch evidential and Dirichlet-based training functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.distributions import Dirichlet
from torch.nn import functional as F
from torch.special import digamma, gammaln

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

# ============================================================================
# Missing utility implementations (stubs to satisfy mypy)
# Replace with your real project implementations later.
# ============================================================================


def normal_wishart_log_prob(
    _m: Tensor,
    _l_precision: Tensor,
    _kappa: Tensor,
    _nu: Tensor,
    mu_k: Tensor,
    _var_k: Tensor,
) -> Tensor:
    """Placeholder Normal-Wishart log-probability implementation.

    Args:
        _m, _l_precision, _kappa, _nu: Unused placeholder parameters.
        mu_k: Mean of ensemble component.
        _var_k: Unused placeholder for variance.

    Returns:
        Zero tensor (placeholder).
    """
    return torch.zeros_like(mu_k)


def make_in_domain_target_alpha(y: Tensor) -> Tensor:
    """Construct target Dirichlet distribution for ID samples.

    Args:
        y: Labels [B].

    Returns:
        Alpha target tensor [B, C].
    """
    num_classes = int(y.max().item()) + 1
    alpha = torch.ones((y.size(0), num_classes), device=y.device)
    alpha[torch.arange(y.size(0)), y] = 10.0
    return alpha


def kl_dirichlet(alpha_p: Tensor, alpha_q: Tensor) -> Tensor:
    """Compute KL(Dir(alpha_p) || Dir(alpha_q)) for each batch item.

    Args:
        alpha_p: Prior Dirichlet parameters.
        alpha_q: Posterior Dirichlet parameters.

    Returns:
        KL divergence [B].
    """
    alpha_p0 = alpha_p.sum(dim=-1, keepdim=True)
    alpha_q0 = alpha_q.sum(dim=-1, keepdim=True)

    term1 = gammaln(alpha_p0) - gammaln(alpha_q0)
    term2 = (gammaln(alpha_q) - gammaln(alpha_p)).sum(dim=-1, keepdim=True)
    term3 = ((alpha_p - alpha_q) * (digamma(alpha_p) - digamma(alpha_p0))).sum(dim=-1, keepdim=True)

    return (term1 + term2 + term3).squeeze(-1)


def predictive_probs(alpha: Tensor) -> Tensor:
    """Expected categorical probabilities under Dirichlet.

    Args:
        alpha: Dirichlet params [B, C].

    Returns:
        Expected probabilities [B, C].
    """
    return alpha / alpha.sum(dim=-1, keepdim=True)


# ============================================================================
# EVIDENTIAL LOSSES
# ============================================================================


class EvidentialLogLoss(nn.Module):
    """Evidential Log Loss from Sensoy et al. (2018)."""

    def __init__(self) -> None:
        """Initialize the evidential log-loss."""
        super().__init__()

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute evidential log loss."""
        alphas = inputs + 1.0
        strengths = alphas.sum(dim=1)
        return torch.mean(torch.log(strengths) - torch.log(alphas[torch.arange(targets.size(0)), targets]))


class EvidentialCELoss(nn.Module):
    """Evidential Cross Entropy Loss."""

    def __init__(self) -> None:
        """Initialize the evidential CE-loss."""
        super().__init__()

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute evidential cross-entropy loss."""
        alphas = inputs + 1.0
        strengths = alphas.sum(dim=1)
        return torch.mean(torch.digamma(strengths) - torch.digamma(alphas[torch.arange(targets.size(0)), targets]))


class EvidentialMSELoss(nn.Module):
    """Evidential Mean Squared Error Loss."""

    def __init__(self) -> None:
        """Initialize evidential MSE-loss."""
        super().__init__()

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute evidential MSE loss."""
        alphas = inputs + 1.0
        strengths = alphas.sum(dim=1)
        y = F.one_hot(targets, inputs.size(1))
        p = alphas / strengths[:, None]

        err = (y - p) ** 2
        var = p * (1 - p) / (strengths[:, None] + 1)

        return torch.mean(torch.sum(err + var, dim=1))


class EvidentialKLDivergence(nn.Module):
    """Evidential KL Divergence Loss."""

    def __init__(self) -> None:
        """Initialize evidential KL-divergence."""
        super().__init__()

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute evidential KL divergence."""
        alphas = inputs + 1.0
        y = F.one_hot(targets, inputs.size(1))
        alphas_tilde = y + (1 - y) * alphas
        strengths_tilde = alphas_tilde.sum(dim=1)

        k = torch.full((inputs.size(0),), inputs.size(1), device=inputs.device)

        first = torch.lgamma(strengths_tilde) - torch.lgamma(k) - torch.sum(torch.lgamma(alphas_tilde), dim=1)
        second = torch.sum(
            (alphas_tilde - 1) * (torch.digamma(alphas_tilde) - torch.digamma(strengths_tilde[:, None])),
            dim=1,
        )

        return torch.mean(first + second)


class EvidentialNIGNLLLoss(nn.Module):
    """Evidence-based NIG regression loss."""

    def __init__(self) -> None:
        """Initialize NIG loss."""
        super().__init__()

    def forward(self, inputs: dict[str, Tensor], targets: Tensor) -> Tensor:
        """Compute NIG negative log-likelihood."""
        omega = 2 * inputs["beta"] * (1 + inputs["nu"])
        return (
            0.5 * torch.log(torch.pi / inputs["nu"])
            - inputs["alpha"] * torch.log(omega)
            + (inputs["alpha"] + 0.5) * torch.log((targets - inputs["gamma"]) ** 2 * inputs["nu"] + omega)
            + torch.lgamma(inputs["alpha"])
            - torch.lgamma(inputs["alpha"] + 0.5)
        ).mean()


class EvidentialRegressionRegularization(nn.Module):
    """Regularization term for evidential regression."""

    def __init__(self) -> None:
        """Initialize regression regularizer."""
        super().__init__()

    def forward(self, inputs: dict[str, Tensor], targets: Tensor) -> Tensor:
        """Compute evidential regression regularization."""
        return (torch.abs(targets - inputs["gamma"]) * (2 * inputs["nu"] + inputs["alpha"])).mean()


# ============================================================================
# NATPN LOSS
# ============================================================================


class NatPNLoss(nn.Module):
    """NatPN classification loss using a Dirichlet-Categorical Bayesian model."""

    def __init__(self, entropy_weight: float = 1e-4) -> None:
        """Initialize NatPN loss."""
        super().__init__()
        self.entropy_weight = entropy_weight

    def forward(self, alpha: Tensor, y: Tensor) -> Tensor:
        """Compute NatPN loss."""
        alpha0 = alpha.sum(dim=-1)
        idx = torch.arange(y.size(0), device=y.device)
        alpha_y = alpha[idx, y]

        expected_nll = torch.digamma(alpha0) - torch.digamma(alpha_y)
        entropy = Dirichlet(alpha).entropy()

        return (expected_nll - self.entropy_weight * entropy).mean()


# ========================================================================================|
# RPN DISTILLATION LOSS
# ========================================================================================|


class RPNDistillationLoss(nn.Module):
    """Regression Prior Network (RPN) distillation loss."""

    def __init__(self) -> None:
        """Initialize RPN distillation loss."""
        super().__init__()

    def forward(
        self,
        rpn_params: tuple[Tensor, Tensor, Tensor, Tensor],
        mus: list[Tensor],
        variances: list[Tensor],
    ) -> Tensor:
        """Compute RPN distillation loss."""
        m, l_precision, kappa, nu = rpn_params
        losses: list[Tensor] = []

        for mu_k, var_k in zip(mus, variances, strict=False):
            logp = normal_wishart_log_prob(m, l_precision, kappa, nu, mu_k, var_k)
            losses.append(-logp.mean())

        return torch.stack(losses).mean()


# ============================================================================
# TRAINING LOOP
# ============================================================================


def dpn_loss_training_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    id_loader: DataLoader,
    device: torch.device,
) -> float:
    """Train one epoch using DPN loss (ID only).

    Args:
        model: A model outputting Dirichlet parameters.
        optimizer: Optimizer instance.
        id_loader: In-distribution DataLoader.
        device: Target device.

    Returns:
        Average epoch loss.
    """
    model.train()

    total_loss = 0.0
    total_batches = 0

    for x_raw, y_raw in id_loader:
        x = x_raw.to(device)
        y = y_raw.to(device)

        optimizer.zero_grad()

        alpha = model(x)
        alpha_target = make_in_domain_target_alpha(y)

        kl = kl_dirichlet(alpha_target, alpha).mean()
        probs = predictive_probs(alpha)
        ce = F.nll_loss(torch.log(probs + 1e-8), y)

        loss = kl + 0.1 * ce
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / total_batches
