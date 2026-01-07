"""Collection of torch evidential and Dirichlet-based training functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.distributions import Dirichlet
from torch.nn import functional as F
from torch.special import digamma, gammaln

if TYPE_CHECKING:
    from probly.layers.evidential import torch as t


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


def make_ood_target_alpha(
    batch_size: int,
    num_classes: int = 10,
    alpha0: float = 10,
) -> torch.Tensor:
    """Construct flat Dirichlet targets for out-of-distribution samples."""
    mu = torch.full(
        (batch_size, num_classes),
        1.0 / num_classes,
        device="cpu",
    )

    return mu * alpha0


def kl_dirichlet(prior_alpha: Tensor, posterior_alpha: Tensor) -> Tensor:
    """Compute KL(Dir(alpha_p) || Dir(alpha_q)) for each batch item.

    Args:
        prior_alpha: Prior Dirichlet parameters.
        posterior_alpha: Posterior Dirichlet parameters.

    Returns:
        KL divergence [B].
    """
    prior_alpha_sum = prior_alpha.sum(dim=-1, keepdim=True)
    posterior_alpha_sum = posterior_alpha.sum(dim=-1, keepdim=True)

    normalization_term = gammaln(prior_alpha_sum) - gammaln(posterior_alpha_sum)
    log_gamma_ratio_term = (gammaln(posterior_alpha) - gammaln(prior_alpha)).sum(dim=-1, keepdim=True)
    digamma_expectation_term = (
        (prior_alpha - posterior_alpha) * (digamma(prior_alpha) - digamma(prior_alpha_sum))
    ).sum(
        dim=-1,
        keepdim=True,
    )

    return (normalization_term + log_gamma_ratio_term + digamma_expectation_term).squeeze(-1)


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


def evidential_log_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    """Evidential Log Loss from Sensoy et al. (2018)."""
    alphas = inputs + 1.0
    strengths = alphas.sum(dim=1)

    loss = torch.mean(torch.log(strengths) - torch.log(alphas[torch.arange(targets.size(0)), targets]))

    return loss


def evidential_ce_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    """Evidential Cross Entropy Loss."""
    alphas = inputs + 1.0
    strengths = alphas.sum(dim=1)

    loss = torch.mean(torch.digamma(strengths) - torch.digamma(alphas[torch.arange(targets.size(0)), targets]))

    return loss


def evidential_mse_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    """Evidential Mean Squared Error Loss."""
    alphas = inputs + 1.0
    strengths = alphas.sum(dim=1)
    y = F.one_hot(targets, inputs.size(1))
    p = alphas / strengths[:, None]

    err = (y - p) ** 2
    var = p * (1 - p) / (strengths[:, None] + 1)

    loss = torch.mean(torch.sum(err + var, dim=1))

    return loss


def evidential_kl_divergence(inputs: Tensor, targets: Tensor) -> Tensor:
    """Evidential KL Divergence Loss."""
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

    loss = torch.mean(first + second)

    return loss


def evidential_nignll_loss(inputs: dict[str, Tensor], targets: Tensor) -> Tensor:
    """Evidence-based NIG regression loss."""
    omega = 2 * inputs["beta"] * (1 + inputs["nu"])
    loss = (
        0.5 * torch.log(torch.pi / inputs["nu"])
        - inputs["alpha"] * torch.log(omega)
        + (inputs["alpha"] + 0.5) * torch.log((targets - inputs["gamma"]) ** 2 * inputs["nu"] + omega)
        + torch.lgamma(inputs["alpha"])
        - torch.lgamma(inputs["alpha"] + 0.5)
    ).mean()

    return loss


def evidential_regression_regularization(inputs: dict[str, Tensor], targets: Tensor) -> Tensor:
    """Regularization term for evidential regression."""
    loss = (torch.abs(targets - inputs["gamma"]) * (2 * inputs["nu"] + inputs["alpha"])).mean()

    return loss


# ========================================================================================|
# RPN DISTILLATION LOSS
# ========================================================================================|


def rpn_distillation_loss(
    rpn_params: tuple[Tensor, Tensor, Tensor, Tensor],
    mus: list[Tensor],
    variances: list[Tensor],
) -> Tensor:
    """Regression Prior Network (RPN) distillation loss."""
    m, l_precision, kappa, nu = rpn_params
    losses: list[Tensor] = []

    for mu_k, var_k in zip(mus, variances, strict=False):
        logp = normal_wishart_log_prob(m, l_precision, kappa, nu, mu_k, var_k)
        losses.append(-logp.mean())

    loss = torch.stack(losses).mean()

    return loss


def postnet_loss(
    z: Tensor,
    y: Tensor,
    flow: t.BatchedRadialFlowDensity,
    class_counts: Tensor,
    entropy_weight: float = 1e-5,
) -> torch.Tensor:
    """Posterior Networks (PostNet) loss."""
    log_dens = flow.log_prob(z)  # [B,C]
    dens = log_dens.exp()

    beta = dens * class_counts.unsqueeze(0)
    alpha = beta + 1.0
    alpha0 = alpha.sum(dim=1)

    digamma = torch.digamma
    batch_idx = torch.arange(len(y), device=y.device)
    expected_ce = digamma(alpha0) - digamma(alpha[batch_idx, y])

    entropy = Dirichlet(alpha).entropy()

    loss = (expected_ce - entropy_weight * entropy).mean()

    return loss, alpha


def lp_fn(alpha: torch.Tensor, y: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    """Compute the Lp calibration loss (upper bound Fi).

    Computes F_i using the expectation-based formulation:
        F_i = ( E[(1-p_c)^p] + Σ_{j≠c} E[p_j^p] )^(1/p)

    Args:
        alpha: Dirichlet concentration parameters, shape (B, K), must be > 0
        y: One-hot encoded labels, shape (B, K)
        p: Lp norm exponent (default: 2.0)

    Returns:
        Scalar loss summed over batch

    Raises:
        ValueError: If alpha contains non-positive values or shapes don't match
    """
    if not torch.all(alpha > 0):
        msg = f"All alpha values must be > 0, got min={alpha.min().item()}"
        raise ValueError(msg)

    if alpha.shape != y.shape:
        msg = f"alpha and y shape mismatch: {alpha.shape} vs {y.shape}"
        raise ValueError(msg)

    # total concentration alpha0
    alpha0 = alpha.sum(dim=1, keepdim=True)  # (B,1)

    # extract alpha_c (correct class)
    alpha_c = (alpha * y).sum(dim=1, keepdim=True)  # (B,1)
    alpha0_minus_c = alpha0 - alpha_c  # (B,1)

    # log B(a,b) used for expectations: E[X^p] = B(a+p,b)/B(a,b)
    def logb(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

    # E[(1 - p_c)^p]   where (1 - p_c) ~ Beta( alpha0 - alpha_c , alpha_c )
    log_e1 = logb(alpha0_minus_c + p, alpha_c) - logb(alpha0_minus_c, alpha_c)
    e1 = torch.exp(log_e1)  # (B,1)

    # Per-class E[p_j^p] for all j
    log_ep = logb(alpha + p, alpha0 - alpha) - logb(alpha, alpha0 - alpha)  # (B,K)
    ep = torch.exp(log_ep)

    # zero-out the true class term so we sum only j≠c
    ep = ep * (1 - y)

    # final expectation sum
    e_sum = e1 + ep.sum(dim=1, keepdim=True)  # (B,1)

    # apply ^(1/p)  # noqa: ERA001
    fi = torch.exp(torch.log(e_sum + 1e-8) / p).squeeze(1)  # (B,)

    loss = fi.sum()

    return loss


def regularization_fn(alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the regularization term using trigamma functions.

    Penalizes high alpha values for incorrect classes to encourage confident
    but calibrated predictions.

    Args:
        alpha: Dirichlet concentration parameters, shape (B, K), must be > 0
        y: One-hot encoded labels, shape (B, K)

    Returns:
        Scalar regularization loss

    Raises:
        ValueError: If shapes don't match
    """
    if alpha.shape != y.shape:
        msg = f"alpha and y shape mismatch: {alpha.shape} vs {y.shape}"
        raise ValueError(msg)

    # Build alpha_tilde by replacing correct-class alpha with 1
    alpha_tilde = alpha * (1 - y) + y

    # Compute alpha_tilde_0 = 1 + sum over incorrect classes
    alpha_tilde_0 = torch.sum(alpha_tilde, dim=1, keepdim=True)

    # Polygamma(1, x) = trigamma(x)
    trigamma_alpha = torch.polygamma(1, alpha_tilde)
    trigamma_alpha0 = torch.polygamma(1, alpha_tilde_0)

    # (alpha_tilde - 1)^2 term
    diff_sq = (alpha_tilde - 1.0) ** 2

    # Penalty only for incorrect classes → mask out true class
    mask = 1 - y

    # Compute elementwise contribution
    term = 0.5 * diff_sq * (trigamma_alpha - trigamma_alpha0) * mask

    # Sum over classes and batch
    loss = torch.sum(term)

    return loss


def dirichlet_entropy(alpha: torch.Tensor) -> torch.Tensor:
    """Compute Dirichlet entropy.

    For adversarial examples, we want to maximize entropy (reward the model for
    being uncertain), which appears as a negative term in the loss.

    Entropy formula:
        H(alpha) = log B(alpha) + (alpha_0 - K) * ψ(alpha_0) - Σ_k (alpha_k - 1) * ψ(alpha_k)

    Args:
        alpha: Dirichlet concentration parameters, shape (B_a, K), must be > 0

    Returns:
        Scalar entropy summed over batch

    Raises:
        ValueError: If alpha contains non-positive values
    """
    if not torch.all(alpha > 0):
        msg = f"All alpha values must be > 0, got min={alpha.min().item()}"
        raise ValueError(msg)

    k = alpha.size(-1)
    alpha0 = alpha.sum(dim=-1)

    log_b = torch.lgamma(alpha).sum(dim=-1) - torch.lgamma(alpha0)

    term1 = log_b
    term2 = (alpha0 - k) * digamma(alpha0)
    term3 = ((alpha - 1) * digamma(alpha)).sum(dim=-1)
    entropy = term1 + term2 - term3

    loss = entropy.sum()

    return loss


def loss_ird(  # noqa: D417
    alpha: torch.Tensor,
    y: torch.Tensor,
    adversarial_alpha: torch.Tensor | None = None,
    p: float = 2.0,
    lam: float = 1.0,
    gamma: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute the Loss introduced in paper: IRD Networks for Predictive Uncertainty Estimation.

    Args:
        alpha : (B, K) Dirichlet concentration parameters
        adversarial_alpha : (B_a, K) adversarial_alpha concentration parameters for adversarial inputs
        y     : (B, K) one-hot labels
        p     : scalar exponent
    Returns:
        loss_ird : the IRD loss comprised of all three terms, summed over all input examples.
    """
    # Input validation
    if alpha.dim() != 2 or y.dim() != 2:
        msg = f"alpha and y must be 2D, got {alpha.dim()}, {y.dim()}"
        raise ValueError(msg)

    if alpha.shape != y.shape:
        msg = f"alpha and y shape mismatch: {alpha.shape} vs {y.shape}"
        raise ValueError(msg)

    if not torch.all(alpha > 0):
        msg = f"All alpha values must be > 0, got min={alpha.min().item()}"
        raise ValueError(msg)

    # Compute Loss Components
    lp_term = lp_fn(alpha, y, p)
    reg_term = regularization_fn(alpha, y)

    if adversarial_alpha is not None:
        if adversarial_alpha.dim() != 2:
            msg = f"adversarial_alpha must be 2D, got {adversarial_alpha.dim()}"
            raise ValueError(msg)

        if adversarial_alpha.shape[1] != alpha.shape[1]:
            msg1 = "adversarial_alpha must have same number of classes as alpha: "
            msg2 = f"{adversarial_alpha.shape[1]} vs {alpha.shape[1]}"
            raise ValueError(msg1 + msg2)

        entropy_term = dirichlet_entropy(adversarial_alpha)
    else:
        entropy_term = 0.0

    # Normalize by batch sizes for stable training across different batch sizes
    if normalize:
        b = alpha.shape[0]
        k = alpha.shape[1]
        lp_term = lp_term / b
        reg_term = reg_term / (b * k)

        if adversarial_alpha is not None and isinstance(entropy_term, torch.Tensor):
            b_a = adversarial_alpha.shape[0]
            entropy_term = entropy_term / b_a

    loss = lp_term + lam * reg_term - gamma * entropy_term

    return loss


def natpn_loss(
    alpha: torch.Tensor,
    y: torch.Tensor,
    entropy_weight: float = 1e-4,
) -> torch.Tensor:
    """NatPN classification loss based on a Dirichlet-Categorical Bayesian formulation.

    Args:
        alpha: Posterior Dirichlet parameters, shape [B, C].
        y: Ground-truth class labels, shape [B] with values in [0, C-1].
        entropy_weight: λ controlling the strength of the entropy regularizer.

    Returns:
        Scalar loss tensor.
    """
    # Total concentration alpha0 per sample
    alpha0 = alpha.sum(dim=-1)  # [B]

    # Digamma function
    digamma = torch.digamma

    # Expected negative log-likelihood for each sample:
    # E[-log p(y)] = ψ(alpha0) - ψ(alpha_y)
    idx = torch.arange(y.size(0), device=y.device)
    expected_nll = digamma(alpha0) - digamma(alpha[idx, y])  # [B]

    # Entropy of Dirichlet posterior
    dir_dist = Dirichlet(alpha)
    entropy = dir_dist.entropy()  # [B]

    loss = (expected_nll - entropy_weight * entropy).mean()

    return loss


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

    loss = (lnll + lam * reg).mean()

    return loss


def rpn_prior(
    shape: torch.Size | tuple[int, ...],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return zero-evidence Normal-Gamma prior parameters for RPN KL."""
    eps = 1e-3
    mu0 = torch.zeros(shape, device=device)
    kappa0 = torch.ones(shape, device=device) * eps
    alpha0 = torch.ones(shape, device=device) * (1.0 + eps)
    beta0 = torch.ones(shape, device=device) * eps

    loss = (mu0, kappa0, alpha0, beta0)

    return loss


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

    loss = (term_mu + term_kappa + term_gamma).mean()

    return loss


def dirichlet_prior_networks_loss(
    alpha_pred: torch.Tensor,
    y: torch.Tensor,
    alpha_ood: torch.Tensor | None = None,
    *,
    ce_weight: float = 0.1,
    num_classes: int | None = None,
) -> torch.Tensor:
    """Implementation of loss-function from Dirichlet Prior Networks: cite:`malininPredictiveUncertaintyEstimation2018`.

    This class implements the Prior Networks framework with dual training on
    in-distribution (ID) and out-of-distribution (OOD) data using KL divergence
    between target and predicted Dirichlet distributions.
    """
    num_classes = num_classes or alpha_pred.shape[1]

    # In-distribution loss: KL(target_sharp || predicted)
    alpha_target_in = make_in_domain_target_alpha(y)
    kl_in = kl_dirichlet(alpha_target_in, alpha_pred).mean()

    # Cross-entropy term for classification stability
    probs_in = predictive_probs(alpha_pred)
    ce_term = F.nll_loss(torch.log(probs_in + 1e-8), y)

    loss = kl_in + ce_weight * ce_term

    # OOD loss: KL(target_flat || predicted)
    if alpha_ood is not None:
        alpha_target_ood = make_ood_target_alpha(
            batch_size=alpha_ood.size(0),
            num_classes=num_classes,
        )
        kl_ood = kl_dirichlet(alpha_target_ood, alpha_ood).mean()
        loss = loss + kl_ood

    return loss


def pn_loss(model: nn.Module, x_in: torch.Tensor, y_in: torch.Tensor, x_ood: torch.Tensor) -> torch.Tensor:
    """Compute the paired ID+OOD loss for one training step.

    This loss is meant to be called inside your training loop after you already sampled
    one in-distribution (ID) mini-batch and one out-of-distribution (OOD) mini-batch.

    Inputs:
      model:  nn.Module
          The network that maps inputs to Dirichlet concentration parameters (alpha).
      x_in:   torch.Tensor
          ID inputs for the current step, shape (B, ...) on the same device as the model.
      y_in:   torch.Tensor
          ID labels for the current step, shape (B,) with class indices (dtype long).
      x_ood:  torch.Tensor
          OOD inputs for the current step, shape (B_ood, ...) on the same device as the model.

    Output:
      torch.Tensor
          A scalar loss tensor (suitable for loss.backward()).
    """
    # ID forward
    alpha_in = model(x_in)
    alpha_target_in = make_in_domain_target_alpha(y_in).to(alpha_in.device)
    kl_in = kl_dirichlet(alpha_target_in, alpha_in).mean()

    probs_in = predictive_probs(alpha_in)
    ce_term = F.nll_loss(torch.log(probs_in + 1e-8), y_in)

    # OOD forward
    alpha_ood = model(x_ood)
    alpha_target_ood = make_ood_target_alpha(x_ood.size(0)).to(alpha_ood.device)
    kl_ood = kl_dirichlet(alpha_target_ood, alpha_ood).mean()

    loss = kl_in + kl_ood + 0.1 * ce_term

    return loss


def rpn_loss(
    model: nn.Module,
    x_id: Tensor,
    y_id: Tensor,
    x_ood: Tensor,
    lam_der: float = 0.01,
    lam_rpn: float = 50.0,
) -> Tensor:
    """Regression Prior Network style loss (Malinin & Gales-ish behavior) for one step with paired ID and OOD batches.

    Inputs:
      model: returns (mu, kappa, alpha, beta) for each x
      x_id:  ID inputs (B_id, ...)
      y_id:  ID targets (B_id, 1) or (B_id,) compatible with der_loss
      x_ood: OOD inputs (B_ood, ...)

    Loss:
      - ID term: DER loss (Student-t NLL + evidence regularizer)
          der_loss(y_id, mu_id, kappa_id, alpha_id, beta_id, lam=lam_der)
      - OOD term: push predicted NIG back to the RPN prior via KL(pred || prior)
          rpn_ng_kl(mu_ood, kappa_ood, alpha_ood, beta_ood, mu0, k0, a0, b0)

      total = loss_id + lam_rpn * loss_ood
    """
    # --- ID forward + supervised DER ---
    mu_id, kappa_id, alpha_id, beta_id = model(x_id)
    loss_id = der_loss(y_id, mu_id, kappa_id, alpha_id, beta_id, lam=lam_der)

    # --- OOD forward + KL to prior (revert to prior / be uninformative) ---
    mu_ood, kappa_ood, alpha_ood, beta_ood = model(x_ood)
    mu0, k0, a0, b0 = rpn_prior(mu_ood.shape, mu_ood.device)

    loss_ood = rpn_ng_kl(mu_ood, kappa_ood, alpha_ood, beta_ood, mu0, k0, a0, b0)

    loss = loss_id + lam_rpn * loss_ood

    return loss
