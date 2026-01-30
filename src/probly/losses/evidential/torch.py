"""Collection of torch evidential and Dirichlet-based training functions."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Dirichlet
from torch.nn import functional as F
from torch.special import digamma, gammaln


def normal_wishart_log_prob(
    _m: Tensor,
    _l_precision: Tensor,
    _kappa: Tensor,
    _nu: Tensor,
    mu_k: Tensor,
    _var_k: Tensor,
) -> Tensor:
    """Placeholder Normal-Wishart log-probability implementation.

    Used by Regression Prior Network (RPN) distillation to compute a
    Normal-Wishart log-likelihood term for ensemble teacher predictions.
    This implementation currently acts as a stub and returns zero-valued
    log-probabilities.

    Args:
        _m: Mean parameter of the Normal-Wishart prior (unused placeholder).
        _l_precision: Precision parameter of the prior (unused placeholder).
        _kappa: Scaling parameter of the prior (unused placeholder).
        _nu: Degrees of freedom of the prior (unused placeholder).
        mu_k: Mean prediction of an ensemble component.
        _var_k: Variance of the ensemble component (unused placeholder).

    Returns:
        Zero tensor (placeholder).
    """
    return torch.zeros_like(mu_k)


def make_in_domain_target_alpha(y: Tensor) -> Tensor:
    """Construct target Dirichlet distribution for in-distribution samples.

    Used by Dirichlet Prior Networks, Posterior Networks, and PN-style paired
    losses to create a sharp (peaked) Dirichlet target for supervised
    in-distribution training.

    Args:
        y: Ground-truth class labels, shape (B,).

    Returns:
        Target Dirichlet concentration parameters, shape (B, C).
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
    """Construct flat Dirichlet target distribution for out-of-distribution samples.

    Used by Dirichlet Prior Networks, Posterior Networks, and PN-style paired
    losses to encourage high uncertainty on out-of-distribution inputs by
    assigning uniform Dirichlet concentration parameters.

    Args:
        batch_size: Number of out-of-distribution samples in the batch.
        num_classes: Number of classes. Defaults to 10.
        alpha0: Total Dirichlet concentration (strength) parameter.

    Returns:
        Target Dirichlet concentration parameters, shape (B, C).
    """
    mu = torch.full(
        (batch_size, num_classes),
        1.0 / num_classes,
    )

    return mu * alpha0


def kl_dirichlet(prior_alpha: Tensor, posterior_alpha: Tensor) -> Tensor:
    """Compute KL(Dir(alpha_p) || Dir(alpha_q)) for each batch item.

    Used by Posterior Networks, Dirichlet Prior Networks, and PN-style
    in-distribution / out-of-distribution losses to compare Dirichlet
    distributions.

    Args:
        prior_alpha: Prior Dirichlet concentration parameters, shape (B, C).
        posterior_alpha: Posterior Dirichlet concentration parameters, shape (B, C).

    Returns:
        KL divergence for each batch element, shape (B,)
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

    Used by Posterior Networks, Dirichlet Prior Networks, and other
    Dirichlet-based classification models to obtain predictive class
    probabilities.

    Args:
        alpha: Dirichlet concentration parameters, shape (B, C).

    Returns:
        Expected categorical probabilities, shape (B, C).
    """
    return alpha / alpha.sum(dim=-1, keepdim=True)


def evidential_log_loss(alphas: Tensor, targets: Tensor) -> Tensor:
    """Evidential Log Loss for classification uncertainty estimation.

    Implements the evidential log loss proposed by Sensoy et al. (2018)
    for Evidential Deep Learning.

    Reference:
        Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty",
        NeurIPS 2018.
        https://arxiv.org/abs/1806.01768

    Args:
        alphas: Dirichlet concentration parameters, shape (B, C).
        targets: Ground-truth class labels, shape (B,).

    Returns:
        Scalar evidential log loss averaged over the batch.
    """
    strengths = alphas.sum(dim=1)

    loss = torch.mean(torch.log(strengths) - torch.log(alphas[torch.arange(targets.size(0)), targets]))

    return loss


def evidential_ce_loss(alphas: Tensor, targets: Tensor) -> Tensor:
    """Evidential Cross Entropy Loss for classification uncertainty estimation.

    Implements the evidential cross-entropy loss proposed by Sensoy et al. (2018)
    for Evidential Deep Learning.

    Reference:
        Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty",
        NeurIPS 2018.
        https://arxiv.org/abs/1806.01768

    Args:
        alphas: Dirichlet concentration parameters, shape (B, C).
        targets: Ground-truth class labels, shape (B,).

    Returns:
        Scalar evidential cross-entropy loss averaged over the batch.
    """
    strengths = alphas.sum(dim=1)

    loss = torch.mean(torch.digamma(strengths) - torch.digamma(alphas[torch.arange(targets.size(0)), targets]))

    return loss


def evidential_mse_loss(alphas: Tensor, targets: Tensor) -> Tensor:
    """Evidential Mean Squared Error loss for classification uncertainty estimation.

    Implements the evidential MSE loss proposed by Sensoy et al. (2018),
    combining prediction error and predictive variance under a Dirichlet
    distribution.

    Reference:
        Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty",
        NeurIPS 2018.
        https://arxiv.org/abs/1806.01768

    Args:
        alphas: Dirichlet concentration parameters, shape (B, C).
        targets: Ground-truth class labels, shape (B,).

    Returns:
        Scalar evidential mean squared error loss averaged over the batch.
    """
    strengths = alphas.sum(dim=1)
    y = F.one_hot(targets, alphas.size(1)).float()
    p = alphas / strengths[:, None]

    err = (y - p) ** 2
    var = p * (1 - p) / (strengths[:, None] + 1)

    loss = torch.mean(torch.sum(err + var, dim=1))

    return loss


def evidential_kl_divergence(alphas: Tensor, targets: Tensor) -> Tensor:
    """Evidential KL divergence loss for classification uncertainty estimation.

    Implements the KL divergence regularization term proposed by
    Sensoy et al. (2018) for Evidential Deep Learning.

    Reference:
        Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty",
        NeurIPS 2018.
        https://arxiv.org/abs/1806.01768

    Args:
        alphas: Dirichlet concentration parameters, shape (B, C).
        targets: Ground-truth class labels, shape (B,).

    Returns:
        Scalar evidential KL divergence loss averaged over the batch.
    """
    y = F.one_hot(targets, alphas.size(1))
    alphas_tilde = y + (1 - y) * alphas
    strengths_tilde = alphas_tilde.sum(dim=1)

    k = torch.full((alphas.size(0),), alphas.size(1), device=alphas.device)

    first = torch.lgamma(strengths_tilde) - torch.lgamma(k) - torch.sum(torch.lgamma(alphas_tilde), dim=1)
    second = torch.sum(
        (alphas_tilde - 1) * (torch.digamma(alphas_tilde) - torch.digamma(strengths_tilde[:, None])),
        dim=1,
    )

    loss = torch.mean(first + second)

    return loss


def evidential_nignll_loss(inputs: dict[str, Tensor], targets: Tensor) -> Tensor:
    """Evidence-based Normal-Inverse-Gamma (NIG) regression loss.

    Implements the negative log-likelihood term used in Deep Evidential
    Regression as proposed by Amini et al. (2020).

    Reference:
        Amini et al., "Deep Evidential Regression",
        NeurIPS 2020.
        https://arxiv.org/abs/1910.02600

    Args:
        inputs: Dictionary containing NIG distribution parameters with keys
            ``"gamma"``, ``"nu"``, ``"alpha"``, and ``"beta"``, each of shape (B,).
        targets: Ground-truth regression targets, shape (B,).

    Returns:
        Scalar NIG negative log-likelihood loss averaged over the batch.
    """
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
    """Regularization term for evidential regression.

    Implements the evidence regularization component proposed by
    Amini et al. (2020) to penalize confident but inaccurate predictions
    in Deep Evidential Regression.

    Reference:
        Amini et al., "Deep Evidential Regression",
        NeurIPS 2020.
        https://arxiv.org/abs/1910.02600

    Args:
        inputs: Dictionary containing evidential regression parameters with keys
            ``"gamma"``, ``"nu"``, and ``"alpha"``, each of shape (B,).
        targets: Ground-truth regression targets, shape (B,).

    Returns:
        Scalar evidential regression regularization loss averaged over the batch.
    """
    loss = (torch.abs(targets - inputs["gamma"]) * (2 * inputs["nu"] + inputs["alpha"])).mean()

    return loss


def rpn_distillation_loss(
    rpn_params: tuple[Tensor, Tensor, Tensor, Tensor],
    mus: list[Tensor],
    variances: list[Tensor],
) -> Tensor:
    """Regression Prior Network (RPN) distillation loss.

    Used in Regression Prior Networks to distill ensemble teacher predictions
    into a student model by minimizing a Normal-Wishart-based divergence term,
    as proposed by Malinin et al. (2020).

    Reference:
        Malinin et al., "Regression Prior Networks",
        NeurIPS 2020.
        https://arxiv.org/abs/2006.11590

    Args:
        rpn_params: Tuple of Normal-Wishart prior parameters
            ``(m, l_precision, kappa, nu)`` produced by the student model.
        mus: List of mean predictions from ensemble teacher models.
        variances: List of predictive variances from ensemble teacher models.

    Returns:
        Scalar RPN distillation loss averaged over ensemble members.
    """
    m, l_precision, kappa, nu = rpn_params
    losses: list[Tensor] = []

    for mu_k, var_k in zip(mus, variances, strict=False):
        logp = normal_wishart_log_prob(m, l_precision, kappa, nu, mu_k, var_k)
        losses.append(-logp.mean())

    loss = torch.stack(losses).mean()

    return loss


def postnet_loss(
    alpha: Tensor,
    y: Tensor,
    entropy_weight: float = 1e-5,
) -> torch.Tensor:
    """Posterior Networks (PostNet) classification loss.

    Implements the expected cross-entropy loss with an entropy regularizer
    as proposed by Charpentier et al. (2020) for Posterior Networks.

    Reference:
        Charpentier et al., "Posterior Networks: Uncertainty Estimation without
        OOD Samples via Density-Based Pseudo-Counts", NeurIPS 2020.
        https://arxiv.org/abs/2006.09239

    Args:
        alpha: Dirichlet concentration parameters, shape (B, C).
        y: Ground-truth class labels, shape (B,).
        entropy_weight: Weight of the entropy regularization term.

    Returns:
        Scalar Posterior Networks loss averaged over the batch.
    """
    alpha0 = alpha.sum(dim=1)
    digamma = torch.digamma
    batch_idx = torch.arange(len(y), device=y.device)
    expected_ce = digamma(alpha0) - digamma(alpha[batch_idx, y])

    entropy = Dirichlet(alpha).entropy()

    loss = (expected_ce - entropy_weight * entropy).mean()

    return loss


def lp_fn(alpha: torch.Tensor, y: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    """Lp calibration loss for predictive uncertainty estimation.

    Implements the Lp calibration loss proposed by Tsiligkaridis (2019) for
    Information Robust Dirichlet Networks.

    Reference:
        Tsiligkaridis, "Information Robust Dirichlet Networks for Predictive Uncertainty Estimation",
        2019.
        https://arxiv.org/abs/1910.04819

    The loss is computed using the expectation-based formulation:
        F_i = ( E[(1 - p_c)^p] + Σ_{j≠c} E[p_j^p] )^(1/p)

    Args:
        alpha: Dirichlet concentration parameters, shape (B, K), must be > 0.
        y: One-hot encoded class labels, shape (B, K).
        p: Lp norm exponent controlling calibration strength (default: 2.0).

    Returns:
        Scalar Lp calibration loss summed over the batch.

    Raises:
        ValueError: If ``alpha`` contains non-positive values or if shapes do not match.
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
    """Regularization term for Information Robust Dirichlet Networks.

    Penalizes high Dirichlet concentration values for incorrect classes to
    encourage confident but well-calibrated predictions.

    Reference:
        Tsiligkaridis, "Information Robust Dirichlet Networks for Predictive Uncertainty Estimation",
        2019.
        https://arxiv.org/abs/1910.04819

    Args:
        alpha: Dirichlet concentration parameters, shape (B, K), must be > 0.
        y: One-hot encoded class labels, shape (B, K).

    Returns:
        Scalar regularization loss summed over classes and batch.

    Raises:
        ValueError: If ``alpha`` and ``y`` shapes do not match.
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
    """Dirichlet entropy for predictive uncertainty estimation.

    Used in Information Robust Dirichlet Networks to encourage uncertainty on
    adversarial or out-of-distribution inputs by maximizing the entropy of the
    Dirichlet distribution.

    Reference:
        Tsiligkaridis, "Information Robust Dirichlet Networks for Predictive Uncertainty Estimation",
        2019.
        https://arxiv.org/abs/1910.04819

    The entropy is given by:
        H(alpha) = log B(alpha)
                   + (alpha_0 - K) * ψ(alpha_0)
                   - Σ_k (alpha_k - 1) * ψ(alpha_k)

    Args:
        alpha: Dirichlet concentration parameters, shape (B_a, K), must be > 0.

    Returns:
        Scalar Dirichlet entropy summed over the batch.

    Raises:
        ValueError: If ``alpha`` contains non-positive values.
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


def loss_ird(
    alpha: torch.Tensor,
    y: torch.Tensor,
    adversarial_alpha: torch.Tensor | None = None,
    p: float = 2.0,
    lam: float = 1.0,
    gamma: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Information Robust Dirichlet (IRD) loss for predictive uncertainty estimation.

    Implements the loss proposed by Tsiligkaridis (2019), combining an
    Lp calibration term, a trigamma-based regularization term, and an
    optional entropy-based adversarial regularizer.

    Reference:
        Tsiligkaridis, "Information Robust Dirichlet Networks for Predictive Uncertainty Estimation",
        2019.
        https://arxiv.org/abs/1910.04819

    Args:
        alpha: Dirichlet concentration parameters, shape (B, K).
        y: One-hot encoded class labels, shape (B, K).
        adversarial_alpha: Dirichlet concentration parameters for adversarial inputs,
            shape (B_a, K).
        p: Lp norm exponent controlling calibration strength.
        lam: Weight of the regularization term.
        gamma: Weight of the entropy regularization term.
        normalize: Whether to normalize loss terms by batch size.

    Returns:
        Scalar IRD loss summed over all input examples.
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
    """Natural Posterior Network (NatPN) classification loss.

    Implements the Dirichlet-Categorical Bayesian loss with an entropy
    regularizer as proposed by Charpentier et al. (2022).

    Reference:
        Charpentier et al., "Natural Posterior Network",
        NeurIPS 2022.
        https://arxiv.org/abs/2105.04471

    Args:
        alpha: Posterior Dirichlet concentration parameters, shape (B, C).
        y: Ground-truth class labels, shape (B,) with values in [0, C-1].
        entropy_weight: Weight controlling the strength of the entropy
            regularization term.

    Returns:
        Scalar NatPN loss averaged over the batch.
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
    """Deep Evidential Regression loss for uncertainty-aware regression.

    Combines a Student-t negative log-likelihood with an evidence
    regularization term as proposed by Amini et al. (2020).

    Reference:
        Amini et al., "Deep Evidential Regression",
        NeurIPS 2020.
        https://arxiv.org/abs/1910.02600

    Args:
        y: Ground-truth regression targets, shape (B,) or (B, 1).
        mu: Predicted mean of the Normal-Inverse-Gamma distribution, shape (B,).
        kappa: Predicted scaling parameter, shape (B,).
        alpha: Predicted shape parameter, shape (B,).
        beta: Predicted scale parameter, shape (B,).
        lam: Weight of the evidence regularization term.

    Returns:
        Scalar Deep Evidential Regression loss averaged over the batch.
    """
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
    """Normal-Gamma prior with zero evidence for Regression Prior Networks.

    Constructs an uninformative Normal-Gamma prior used in Regression Prior
    Networks to regularize out-of-distribution predictions via KL divergence,
    as proposed by Malinin et al. (2020).

    Reference:
        Malinin et al., "Regression Prior Networks",
        NeurIPS 2020.
        https://arxiv.org/abs/2006.11590

    Args:
        shape: Shape of the prior parameter tensors (e.g., batch shape).
        device: Torch device on which to allocate the tensors.

    Returns:
        Tuple ``(mu0, kappa0, alpha0, beta0)`` of Normal-Gamma prior parameters,
        each with the specified shape.
    """
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
    """KL divergence between two Normal-Gamma distributions.

    Computes the KL divergence between a predicted Normal-Gamma distribution
    and a prior Normal-Gamma distribution, as used in Regression Prior Networks
    to regularize out-of-distribution predictions.

    Reference:
        Malinin et al., "Regression Prior Networks",
        NeurIPS 2020.
        https://arxiv.org/abs/2006.11590

    Args:
        mu: Predicted mean parameter, shape (B,).
        kappa: Predicted scaling parameter, shape (B,).
        alpha: Predicted shape parameter, shape (B,).
        beta: Predicted scale parameter, shape (B,).
        mu0: Prior mean parameter, shape (B,).
        kappa0: Prior scaling parameter, shape (B,).
        alpha0: Prior shape parameter, shape (B,).
        beta0: Prior scale parameter, shape (B,).

    Returns:
        Scalar KL divergence between predicted and prior Normal-Gamma
        distributions, averaged over the batch.
    """
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
    """Dirichlet Prior Networks (DPN) loss for uncertainty-aware classification.

    Implements the training objective proposed by Malinin and Gales (2018),
    combining KL divergence to sharp in-distribution targets, optional KL
    divergence to flat out-of-distribution targets, and a cross-entropy
    stabilizing term.

    Reference:
        Malinin and Gales, "Predictive Uncertainty Estimation via Prior Networks",
        NeurIPS 2018.
        https://arxiv.org/abs/1802.10501

    Args:
        alpha_pred: Predicted Dirichlet concentration parameters, shape (B, C).
        y: Ground-truth class labels, shape (B,).
        alpha_ood: Predicted Dirichlet parameters for out-of-distribution inputs,
            shape (B_ood, C).
        ce_weight: Weight of the cross-entropy stabilizer term.
        num_classes: Number of classes. If None, inferred from ``alpha_pred``.

    Returns:
        Scalar Dirichlet Prior Networks loss.
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
    """Paired in-distribution and out-of-distribution loss for Posterior Networks.

    Computes the combined loss for one training step using paired
    in-distribution (ID) and out-of-distribution (OOD) mini-batches, as
    proposed by Charpentier et al. (2020).

    Reference:
        Charpentier et al., "Posterior Networks: Uncertainty Estimation without
        OOD Samples via Density-Based Pseudo-Counts", NeurIPS 2020.
        https://arxiv.org/abs/2006.09239

    Args:
        model: Network mapping inputs to Dirichlet concentration parameters.
        x_in: In-distribution inputs for the current step, shape (B, ...).
        y_in: In-distribution class labels, shape (B,).
        x_ood: Out-of-distribution inputs for the current step, shape (B_ood, ...).

    Returns:
        Scalar paired ID+OOD Posterior Networks loss.
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
    """Paired in-distribution and out-of-distribution loss for Regression Prior Networks.

    Computes the Regression Prior Network (RPN) training objective using
    paired in-distribution (ID) and out-of-distribution (OOD) mini-batches.
    The loss combines a supervised Deep Evidential Regression (DER) term
    on ID data with a KL regularization term that pushes OOD predictions
    back toward the Normal-Gamma prior.

    Reference:
        Malinin et al., "Regression Prior Networks",
        NeurIPS 2020.
        https://arxiv.org/abs/2006.11590

    Args:
        model: Regression model returning (mu, kappa, alpha, beta) for each input.
        x_id: In-distribution inputs, shape (B_id, ...).
        y_id: In-distribution regression targets, shape (B_id,) or compatible.
        x_ood: Out-of-distribution inputs, shape (B_ood, ...).
        lam_der: Weight of the DER evidence regularization term.
        lam_rpn: Weight of the RPN prior-matching KL term.

    Returns:
        Scalar paired ID+OOD Regression Prior Network loss.
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
