"""Torch training losses."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.distributions import Dirichlet, kl_divergence
from torch.nn import functional as F
from torch.special import digamma

from probly.layers.torch import (
    GVBLLLayer,
    HetVBLLLayer,
    TVBLLLayer,
    VBLLLayer,
    VBLLParameterization,
)
from probly.utils.torch import dirichlet_entropy, intersection_probability

from ._common import vbll_loss

# --- Classification ----------------------------------------------------------


def label_relaxation_loss(inputs: torch.Tensor, targets: torch.Tensor, *, alpha: float = 0.1) -> torch.Tensor:
    """Label Relaxation Loss from :cite:`lienenFromLabel2021`.

    This loss is used to improve the calibration of a neural network. It works by minimizing
    the Kullback-Leibler divergence between the predicted probabilities and the target distribution in the credal set
    defined by the alpha parameter. The target distribution is the distribution in the credal set that minimizes the
    Kullback-Leibler divergence from the predicted probabilities. If the predicted probability distribution
    is in the credal set, the loss is zero.

    Args:
        inputs: Logits of size (n_instances, n_classes).
        targets: Class labels of size (n_instances,).
        alpha: The parameter that controls the amount of label relaxation. Increasing alpha, increases the size
            of the credal set and thus the amount of label relaxation.

    Returns:
        The mean loss value.
    """
    inputs_probs = F.softmax(inputs, dim=1)

    with torch.no_grad():
        inv_one_hot = 1 - F.one_hot(targets, inputs.shape[1])
        targets_real = alpha * inputs_probs / torch.sum(inv_one_hot * inputs_probs, dim=1, keepdim=True)
        targets_real[torch.arange(targets.shape[0]), targets] = 1 - alpha

    kl_div = torch.sum(F.kl_div(inputs_probs.log(), targets_real, log_target=False, reduction="none"), dim=1)
    loss = torch.where(torch.sum(inv_one_hot * inputs_probs, dim=1) <= alpha, 0, kl_div)
    return loss.mean()


def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, *, alpha: float = 1, gamma: float = 2) -> torch.Tensor:
    """Focal Loss based on :cite:`linFocalLoss2017`.

    Args:
        inputs: Logits of size (n_instances, n_classes).
        targets: Class labels of size (n_instances,).
        alpha: Control importance of minority class.
        gamma: Control loss for hard instances.

    Returns:
        The mean loss value.
    """
    targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[-1])
    prob = F.softmax(inputs, dim=-1)
    p_t = torch.sum(prob * targets_one_hot, dim=-1)

    log_prob = torch.log(prob)
    loss = -alpha * (1 - p_t) ** gamma * torch.sum(log_prob * targets_one_hot, dim=-1)

    return torch.mean(loss)


def cvar_ce_loss(output: Tensor, targets: Tensor, delta: float) -> Tensor:
    """CVaR cross-entropy loss from :cite:`wangLearningCredalEnsembles2026`.

    The batch-wise CVaR approximation of Eq. 7 averages cross-entropy over the
    top ``floor(delta * B)`` highest-loss samples: only the worst ``delta``
    fraction of the batch receives gradient. ``delta=1`` recovers the batch mean (ERM).

    Args:
        output: Logits of shape ``(B, num_classes)``.
        targets: Ground-truth class indices of shape ``(B,)``.
        delta: Fraction of highest-loss samples to keep, in (0, 1].

    Returns:
        Scalar cross-entropy loss averaged over the selected samples.

    Raises:
        ValueError: If delta is outside (0, 1].
    """
    if not 0.0 < delta <= 1.0:
        msg = f"delta must be in (0, 1], got {delta}."
        raise ValueError(msg)
    per_sample = F.cross_entropy(output, targets, reduction="none")
    if delta >= 1.0:
        return per_sample.mean()
    # floor(delta * B), clamped to 1 so degenerate tiny batches still train.
    k = max(1, int(delta * per_sample.shape[0]))
    return per_sample.topk(k).values.mean()


def intersection_probability_ce_loss(output: Tensor, targets: Tensor) -> Tensor:
    """Intersection-probability cross-entropy loss from :cite:`wangCredalDeepEnsembles2024`.

    Implements Eq. 14 for interval-valued predictions. Splits the packed
    ``(B, 2C)`` interval output into ``(lower, upper)``, computes the
    intersection probability, and applies negative-log-likelihood against
    the targets. The probabilities are clamped to ``finfo(dtype).eps``
    before the log to avoid ``-inf``.

    Args:
        output: Packed ``(B, 2 * num_classes)`` tensor with the lower bounds
            in the first half and the upper bounds in the second.
        targets: Ground-truth class indices of shape ``(B,)``.

    Returns:
        Scalar cross-entropy loss averaged over the batch.
    """
    n_classes = output.shape[-1] // 2
    q_int = intersection_probability(output[..., :n_classes], output[..., n_classes:])
    eps = torch.finfo(q_int.dtype).eps
    return F.nll_loss(torch.log(q_int.clamp(min=eps)), targets)


# --- Variational -------------------------------------------------------------


def elbo_loss(
    inputs: torch.Tensor, targets: torch.Tensor, kl: torch.Tensor, *, kl_penalty: float = 1e-5
) -> torch.Tensor:
    """Evidence lower bound loss based on :cite:`blundellWeightUncertainty2015`.

    Args:
        inputs: Logits of size (n_instances, n_classes).
        targets: Class labels of size (n_instances,).
        kl: KL divergence of the model.
        kl_penalty: Weight for KL divergence term.

    Returns:
        The mean loss value.
    """
    return F.cross_entropy(inputs, targets) + kl_penalty * kl


def _gaussian_weight_kl(
    mean: torch.Tensor,
    logdiag: torch.Tensor,
    offdiag: torch.Tensor | None,
    parameterization: VBLLParameterization,
    prior_scale: float,
    cov_factor: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Expected KL from a Gaussian weight posterior to an isotropic prior ``N(0, prior_scale * I)``.

    Implements the precision-weighted ``expected_gaussian_kl`` of the reference
    VBLL implementation: the squared-mean term is scaled by ``cov_factor`` (an
    expected noise precision), which may be a scalar or broadcast per-class /
    per-sample tensor.

    Args:
        mean: Posterior mean, shape ``(num_classes, in_features)``.
        logdiag: Log Cholesky diagonal, shape ``(num_classes, in_features)``.
        offdiag: Strict-lower Cholesky entries (dense) or ``None`` (diagonal).
        parameterization: ``"diagonal"`` or ``"dense"``.
        prior_scale: Scale of the isotropic prior covariance.
        cov_factor: Per-class (or per-sample) weighting of the squared-mean term.

    Returns:
        The summed KL, reduced over classes; shape depends on ``cov_factor`` broadcasting.
    """
    in_features = mean.shape[-1]
    mean_sq = mean.square().sum(dim=-1) / prior_scale
    combined_mean_sq = (cov_factor * mean_sq).sum(dim=-1)
    if parameterization == "diagonal":
        trace = torch.exp(2.0 * logdiag).sum(dim=-1)
    else:
        chol = torch.tril(cast("torch.Tensor", offdiag), diagonal=-1) + torch.diag_embed(torch.exp(logdiag))
        trace = chol.square().sum(dim=(-2, -1))
    trace_term = (trace / prior_scale).sum(dim=-1)
    log_det_term = (in_features * math.log(prior_scale) - 2.0 * logdiag.sum(dim=-1)).sum(dim=-1)
    return 0.5 * (combined_mean_sq + trace_term + log_det_term)


def _reduced_kn_bound(
    mean: torch.Tensor,
    cov: torch.Tensor,
    targets: torch.Tensor,
    alpha: torch.Tensor,
    expected_cov: torch.Tensor,
    expected_prec: torch.Tensor,
) -> torch.Tensor:
    """Reduced Knowles-Minka lower bound on the expected softmax log-likelihood.

    Shared by the Student-t and heteroscedastic objectives; the two differ only
    in how the expected noise covariance and precision are obtained.

    Args:
        mean: Logit means, shape ``(batch, num_classes)``.
        cov: Weight-posterior logit variance plus one, shape ``(batch, num_classes)``.
        targets: Integer class labels, shape ``(batch,)``.
        alpha: Learnable coefficient of the bound.
        expected_cov: Expected noise covariance, broadcastable to ``cov``.
        expected_prec: Expected noise precision, broadcastable to ``cov``.

    Returns:
        The per-sample bound, shape ``(batch,)``.
    """
    index = torch.arange(mean.shape[0])
    linear_term = mean[index, targets]
    lse_term = torch.logsumexp(mean + alpha * cov, dim=-1)
    cov_term = cov * (expected_cov / 4.0 + expected_prec * alpha**2 - alpha)
    return linear_term - lse_term - 0.5 * cov_term.sum(dim=-1)


@vbll_loss.register(VBLLLayer)
def disc_vbll_loss(
    layer: VBLLLayer,
    features: torch.Tensor,
    targets: torch.Tensor,
    regularization_weight: float,
) -> torch.Tensor:
    """Negative discriminative VBLL ELBO from :cite:`harrisonVariationalBayesian2024`.

    Implements the discriminative classification objective of a
    :class:`~probly.layers.torch.VBLLLayer`: the closed-form double-Jensen lower
    bound on the expected log-likelihood, regularized by the weight-posterior
    :attr:`~probly.layers.torch.VBLLLayer.kl_divergence` and a Wishart term on
    the learnable noise precision. Both ingredients of the bound - the logit
    mean and the logit variance ``phi^T S_k phi + sigma_k^2`` - are exactly the
    ``(mean, var)`` returned by the layer's forward pass.

    Args:
        layer: The variational Bayesian last layer to fit.
        features: Backbone features feeding the layer, shape ``(batch, in_features)``.
        targets: Integer class labels, shape ``(batch,)``.
        regularization_weight: Weight on the regularization terms (typically
            ``1 / dataset_size``).

    Returns:
        A scalar tensor with the negative ELBO to minimize.
    """
    mean, var = layer(features)
    index = torch.arange(features.shape[0])
    true_logit = mean[index, targets]
    log_normalizer = torch.logsumexp(mean + 0.5 * var, dim=-1)
    expected_log_likelihood = (true_logit - log_normalizer).mean()

    total_elbo = expected_log_likelihood + regularization_weight * (layer.noise_wishart_term - layer.kl_divergence)
    return -total_elbo


@vbll_loss.register(GVBLLLayer)
def g_vbll_loss(
    layer: GVBLLLayer,
    features: torch.Tensor,
    targets: torch.Tensor,
    regularization_weight: float,
) -> torch.Tensor:
    """Negative generative VBLL ELBO from :cite:`harrisonVariationalBayesian2024`.

    Implements the discriminative-free generative training objective of
    a :class:`~probly.layers.torch.GVBLLLayer`: the Jensen lower bound on the expected
    class-conditional log-likelihood, plus the class-mean KL term and a Wishart
    term on the shared noise precision.

    Args:
        layer: The generative variational Bayesian last layer to fit.
        features: Backbone features feeding the layer, shape ``(batch, in_features)``.
        targets: Integer class labels, shape ``(batch,)``.
        regularization_weight: Weight on the regularization terms (typically
            ``1 / dataset_size``).

    Returns:
        A scalar tensor with the negative ELBO to minimize.
    """
    noise_log_var = 2.0 * layer.noise_logdiag
    noise_var = torch.exp(noise_log_var)

    mu_target = layer.mu_mean[targets]
    diff = features - mu_target
    linear_term = -0.5 * ((diff.square() / noise_var) + noise_log_var + math.log(2.0 * math.pi)).sum(dim=-1)

    trace_term = (torch.exp(2.0 * layer.mu_logdiag[targets]) / noise_var).sum(dim=-1)
    lse_term = torch.logsumexp(layer(features), dim=-1)
    jensen_bound = linear_term - 0.5 * trace_term - lse_term

    total_elbo = jensen_bound.mean() + regularization_weight * (layer.noise_wishart_term - layer.kl_divergence)
    return -total_elbo


@vbll_loss.register(TVBLLLayer)
def t_vbll_loss(
    layer: TVBLLLayer,
    features: torch.Tensor,
    targets: torch.Tensor,
    regularization_weight: float,
) -> torch.Tensor:
    """Negative Student-t VBLL ELBO from :cite:`harrisonVariationalBayesian2024`.

    Implements the Student-t discriminative objective of
    a :class:`~probly.layers.torch.TVBLLLayer`, combining the reduced Knowles-Minka
    softmax bound with the Gamma noise-precision KL and the weight-posterior KL.

    Args:
        layer: The Student-t variational Bayesian last layer to fit.
        features: Backbone features feeding the layer, shape ``(batch, in_features)``.
        targets: Integer class labels, shape ``(batch,)``.
        regularization_weight: Weight on the regularization terms (typically ``1 / dataset_size``).

    Returns:
        A scalar tensor with the negative ELBO to minimize.
    """
    mean, weight_variance = layer.logit_moments(features)
    cov = weight_variance + 1.0

    expected_cov = torch.exp(layer.noise_log_rate - layer.noise_log_dof + 1.0)
    expected_prec = torch.exp(layer.noise_log_dof - layer.noise_log_rate)
    bound = _reduced_kn_bound(mean, cov, targets, layer.alpha, expected_cov, expected_prec)

    gamma_kl = torch.distributions.kl_divergence(layer.noise, layer.noise_prior).sum(dim=-1)
    weight_kl = _gaussian_weight_kl(
        layer.W_mean,
        layer.W_logdiag,
        layer.offdiag(),
        layer.parameterization,
        layer.prior_scale,
        expected_prec,
    )

    total_elbo = bound.mean() - regularization_weight * (gamma_kl + weight_kl)
    return -total_elbo


@vbll_loss.register(HetVBLLLayer)
def het_vbll_loss(
    layer: HetVBLLLayer,
    features: torch.Tensor,
    targets: torch.Tensor,
    regularization_weight: float,
) -> torch.Tensor:
    """Negative heteroscedastic VBLL ELBO from :cite:`harrisonVariationalBayesian2024`.

    Implements the heteroscedastic discriminative objective of
    a :class:`~probly.layers.torch.HetVBLLLayer`, combining the reduced Knowles-Minka
    softmax bound with the input-dependent noise KL and the weight-posterior KL.

    Args:
        layer: The heteroscedastic variational Bayesian last layer to fit.
        features: Backbone features feeding the layer, shape ``(batch, in_features)``.
        targets: Integer class labels, shape ``(batch,)``.
        regularization_weight: Weight on the regularization terms (typically ``1 / dataset_size``).

    Returns:
        A scalar tensor with the negative ELBO to minimize.
    """
    mean, weight_variance = layer.logit_moments(features)
    cov = weight_variance + 1.0

    log_noise_mean, log_noise_var = layer.log_noise_moments(features)
    expected_cov = torch.exp(log_noise_mean + 0.5 * log_noise_var)
    expected_prec = torch.exp(-log_noise_mean + 0.5 * log_noise_var)
    bound = _reduced_kn_bound(mean, cov, targets, layer.alpha, expected_cov, expected_prec)

    weight_kl = _gaussian_weight_kl(
        layer.W_mean,
        layer.W_logdiag,
        layer.w_offdiag(),
        layer.parameterization,
        layer.prior_scale,
        expected_prec,
    ).mean()
    noise_kl = _gaussian_weight_kl(
        layer.M_mean,
        layer.M_logdiag,
        layer.m_offdiag(),
        layer.parameterization,
        layer.noise_prior_scale,
    )

    total_elbo = bound.mean() - regularization_weight * (weight_kl + noise_kl)
    return -total_elbo


# --- Evidential --------------------------------------------------------------


def make_in_domain_target_alpha(y: Tensor) -> Tensor:
    """Construct in-domain Dirichlet targets based on :cite:`malininPredictiveUncertaintyEstimation2018`.

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
    """Construct OOD Dirichlet targets based on :cite:`malininPredictiveUncertaintyEstimation2018`.

    Used by Dirichlet Prior Networks, Posterior Networks, and PN-style paired
    losses to encourage high uncertainty on out-of-distribution inputs by
    assigning uniform Dirichlet concentration parameters.

    Args:
        batch_size: Number of out-of-distribution samples in the batch.
        num_classes: Number of classes. Defaults to 10.
        alpha0: Total Dirichlet concentration parameter (strength).

    Returns:
        Target Dirichlet concentration parameters, shape (B, C).
    """
    mu = torch.full(
        (batch_size, num_classes),
        1.0 / num_classes,
    )

    return mu * alpha0


def evidential_log_loss(alphas: Tensor, targets: Tensor) -> Tensor:
    """Evidential log loss from :cite:`sensoyEvidentialDeep2018`.

    Implements the evidential log loss for classification uncertainty estimation
    in Evidential Deep Learning.

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
    """Evidential cross-entropy loss from :cite:`sensoyEvidentialDeep2018`.

    Implements the evidential cross-entropy loss for classification uncertainty
    estimation in Evidential Deep Learning.

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
    """Evidential mean squared error loss from :cite:`sensoyEvidentialDeep2018`.

    Implements the evidential MSE loss for classification uncertainty estimation,
    combining prediction error and predictive variance under a Dirichlet
    distribution.

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
    """Evidential KL divergence regularizer from :cite:`sensoyEvidentialDeep2018`.

    Implements the KL divergence regularization term for classification
    uncertainty estimation in Evidential Deep Learning.

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
    """Evidential Normal-Inverse-Gamma regression loss from :cite:`aminiDeepEvidential2020`.

    Implements the negative log-likelihood term used in Deep Evidential
    Regression with a Normal-Inverse-Gamma (NIG) distribution.

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
    """Evidential regression regularizer from :cite:`aminiDeepEvidential2020`.

    Implements the evidence regularization component to penalize confident but
    inaccurate predictions in Deep Evidential Regression.

    Args:
        inputs: Dictionary containing evidential regression parameters with keys
            ``"gamma"``, ``"nu"``, and ``"alpha"``, each of shape (B,).
        targets: Ground-truth regression targets, shape (B,).

    Returns:
        Scalar evidential regression regularization loss averaged over the batch.
    """
    loss = (torch.abs(targets - inputs["gamma"]) * (2 * inputs["nu"] + inputs["alpha"])).mean()

    return loss


def pn_loss(model: nn.Module, x_in: torch.Tensor, y_in: torch.Tensor, x_ood: torch.Tensor) -> torch.Tensor:
    """Dirichlet Prior Network loss based on :cite:`malininPredictiveUncertaintyEstimation2018`.

    Combines KL divergence to sharp in-distribution targets and flat
    out-of-distribution targets, with an additional cross-entropy term for
    classification stability.

    Args:
        model: Network mapping inputs to Dirichlet concentration parameters.
        x_in: In-distribution inputs, shape (B, ...).
        y_in: In-distribution class labels, shape (B,).
        x_ood: Out-of-distribution inputs, shape (B_ood, ...).

    Returns:
        Scalar paired ID+OOD Prior Networks loss.
    """
    # ID forward
    alpha_in = model(x_in)
    alpha_target_in = make_in_domain_target_alpha(y_in).to(alpha_in.device)
    kl_in = kl_divergence(
        Dirichlet(alpha_target_in, validate_args=False), Dirichlet(alpha_in, validate_args=False)
    ).mean()

    probs_in = alpha_in / alpha_in.sum(dim=-1, keepdim=True)
    ce_term = F.nll_loss(torch.log(probs_in + 1e-8), y_in)

    # OOD forward
    alpha_ood = model(x_ood)
    alpha_target_ood = make_ood_target_alpha(x_ood.size(0)).to(alpha_ood.device)
    kl_ood = kl_divergence(
        Dirichlet(alpha_target_ood, validate_args=False), Dirichlet(alpha_ood, validate_args=False)
    ).mean()

    loss = kl_in + kl_ood + 0.1 * ce_term

    return loss


def postnet_loss(
    alpha: Tensor,
    y: Tensor,
    entropy_weight: float = 1e-5,
    reduction: str = "sum",
) -> torch.Tensor:
    """Posterior Network classification loss from :cite:`charpentierPosteriorNetwork2020`.

    Implements the expected cross-entropy loss with an entropy regularizer
    for Posterior Networks (PostNet).

    Args:
        alpha: Dirichlet concentration parameters, shape (B, C).
        y: Ground-truth class labels, shape (B,).
        entropy_weight: Weight of the entropy regularization term. Defaults to 1e-5 as used in the original paper.
        reduction: Specifies the reduction to apply to the output. Can be 'mean' or 'sum'. Defaults to 'sum'
            to align with the implementation in the original paper.

    Returns:
        Scalar Posterior Networks loss averaged over the batch.
    """
    alpha0 = alpha.sum(dim=1)
    batch_idx = torch.arange(y.shape[0], device=y.device)
    cross_entropy = digamma(alpha0) - digamma(alpha[batch_idx, y])
    entropy = Dirichlet(alpha).entropy()
    if reduction == "mean":
        loss = (cross_entropy - entropy_weight * entropy).mean()
    elif reduction == "sum":
        loss = (cross_entropy - entropy_weight * entropy).sum()
    return loss


def mixture_uce_loss(
    alpha: torch.Tensor,
    mixture_weights: torch.Tensor,
    y: torch.Tensor,
    reduction: str = "sum",
) -> torch.Tensor:
    """LOP-GPN mixture uncertainty cross-entropy loss from :cite:`damkeLinearOpinionPooling2024`.

    By linearity of expectation, the expected cross-entropy under a Dirichlet
    mixture is the weighted sum of the component Dirichlet cross-entropies.

    Args:
        alpha: Feature-level Dirichlet concentration parameters with shape ``(N, C)``.
        mixture_weights: Dense mixture weights with shape ``(B, N)``.
        y: Ground-truth labels for the mixed nodes with shape ``(B,)``.
        reduction: Reduction to apply, either ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Mixture uncertainty cross-entropy loss.

    Raises:
        ValueError: If ``reduction`` is unsupported.
    """
    alpha_sum = alpha.sum(dim=-1)
    mixture_sum_digamma = mixture_weights @ digamma(alpha_sum).view(-1, 1)
    mixture_digamma = mixture_weights @ digamma(alpha)
    batch_idx = torch.arange(y.shape[0], device=y.device)
    loss = mixture_sum_digamma.squeeze(-1) - mixture_digamma[batch_idx, y]
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    msg = f"Unsupported reduction: {reduction!r}."
    raise ValueError(msg)


def lop_gpn_loss(
    alpha_features: torch.Tensor,
    mixture_weights: torch.Tensor,
    y: torch.Tensor,
    entropy_regularization: torch.Tensor | None = None,
    entropy_weight: float = 0.0,
    reduction: str = "sum",
) -> torch.Tensor:
    """LOP-GPN loss based on :cite:`damkeLinearOpinionPooling2024`.

    Uses the mixture UCE objective computed by :func:`mixture_uce_loss`,
    with an optional caller-supplied entropy regularizer.

    Args:
        alpha_features: Feature-level Dirichlet concentration parameters with shape ``(N, C)``.
        mixture_weights: Dense mixture weights with shape ``(B, N)``.
        y: Ground-truth labels for the mixed nodes with shape ``(B,)``.
        entropy_regularization: Optional per-sample entropy regularizer.
        entropy_weight: Weight applied to ``entropy_regularization``.
        reduction: Reduction to apply, either ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        Scalar or per-sample LOP-GPN loss.
    """
    loss = mixture_uce_loss(alpha_features, mixture_weights, y, reduction=reduction)
    if entropy_regularization is None or entropy_weight == 0.0:
        return loss
    if reduction == "mean":
        return loss - entropy_weight * entropy_regularization.mean()
    if reduction == "sum":
        return loss - entropy_weight * entropy_regularization.sum()
    if reduction == "none":
        return loss - entropy_weight * entropy_regularization
    msg = f"Unsupported reduction: {reduction!r}."
    raise ValueError(msg)


def natpn_loss(
    alpha: torch.Tensor,
    y: torch.Tensor,
    entropy_weight: float = 1e-4,
) -> torch.Tensor:
    """Natural Posterior Network loss from :cite:`charpentierNaturalPosteriorNetwork2022`.

    Implements the Dirichlet-Categorical Bayesian loss with an entropy
    regularizer for Natural Posterior Network (NatPN) classification.

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


def ird_loss(
    alpha: torch.Tensor,
    y: torch.Tensor,
    adversarial_alpha: torch.Tensor | None = None,
    p: float = 2.0,
    lam: float = 1.0,
    gamma: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Information Robust Dirichlet loss from :cite:`tsiligkaridisInformationRobustDirichlet2019`.

    Implements the Information Robust Dirichlet (IRD) loss, combining an
    Lp calibration term, a trigamma-based regularization term, and an
    optional entropy-based adversarial regularizer.

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

        if not torch.all(adversarial_alpha > 0):
            msg = f"All alpha values must be > 0, got min={adversarial_alpha.min().item()}"
            raise ValueError(msg)

        entropy_term = dirichlet_entropy(adversarial_alpha).sum()
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


def lp_fn(alpha: torch.Tensor, y: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    """Lp calibration loss from :cite:`tsiligkaridisInformationRobustDirichlet2019`.

    Implements the Lp calibration loss for predictive uncertainty estimation
    in Information Robust Dirichlet Networks.

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
    """Information Robust Dirichlet regularizer from :cite:`tsiligkaridisInformationRobustDirichlet2019`.

    Penalizes high Dirichlet concentration values for incorrect classes to
    encourage confident but well-calibrated predictions.

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


def der_loss(
    y: Tensor,
    mu: Tensor,
    kappa: Tensor,
    alpha: Tensor,
    beta: Tensor,
    lam: float = 0.01,
) -> Tensor:
    """Deep Evidential Regression loss from :cite:`aminiDeepEvidential2020`.

    Combines a Student-t negative log-likelihood with an evidence
    regularization term for uncertainty-aware regression.

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


def rpn_loss(
    model: nn.Module,
    x_id: Tensor,
    y_id: Tensor,
    x_ood: Tensor,
    lam_der: float = 0.01,
    lam_rpn: float = 50.0,
) -> Tensor:
    """Paired ID/OOD Regression Prior Network loss based on :cite:`malininRegressionPriorNetworks2020`.

    Computes a Regression Prior Network (RPN) training objective using paired in-distribution (ID)
    and out-of-distribution (OOD) mini-batches.
    The loss combines a supervised Deep Evidential Regression (DER) term
    on ID data with a KL regularization term that pushes OOD predictions
    back toward the Normal-Gamma prior.

    Args:
        model: Regression model returning a dict with the keys "gamma", "nu", "alpha" and "beta", as produced
            by :func:`probly.method.evidential.evidential_regression`.
        x_id: In-distribution inputs, shape (B_id, ...).
        y_id: In-distribution regression targets, shape (B_id,) or compatible.
        x_ood: Out-of-distribution inputs, shape (B_ood, ...).
        lam_der: Weight of the DER evidence regularization term.
        lam_rpn: Weight of the RPN prior-matching KL term.

    Returns:
        Scalar paired ID+OOD Regression Prior Network loss.
    """
    # --- ID forward + supervised DER ---
    out_id = model(x_id)
    loss_id = der_loss(y_id, out_id["gamma"], out_id["nu"], out_id["alpha"], out_id["beta"], lam=lam_der)

    # --- OOD forward + KL to prior (revert to prior / be uninformative) ---
    out_ood = model(x_ood)
    mu0, k0, a0, b0 = rpn_prior(out_ood["gamma"].shape, out_ood["gamma"].device)

    loss_ood = rpn_ng_kl(out_ood["gamma"], out_ood["nu"], out_ood["alpha"], out_ood["beta"], mu0, k0, a0, b0)

    loss = loss_id + lam_rpn * loss_ood

    return loss


def rpn_prior(
    shape: torch.Size | tuple[int, ...],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Normal-Gamma prior for Regression Prior Networks from :cite:`malininRegressionPriorNetworks2020`.

    Constructs an uninformative Normal-Gamma prior used in Regression Prior
    Networks to regularize out-of-distribution predictions via KL divergence.

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
    """Normal-Gamma KL divergence for Regression Prior Networks from :cite:`malininRegressionPriorNetworks2020`.

    Computes the KL divergence between a predicted Normal-Gamma distribution
    and a prior Normal-Gamma distribution, as used in Regression Prior Networks
    to regularize out-of-distribution predictions.

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


def normal_wishart_log_prob(
    m: Tensor,
    l_precision: Tensor,
    kappa: Tensor,
    nu: Tensor,
    mu_k: Tensor,
    sigma2_k: Tensor,
) -> Tensor:
    """Simplified Normal-Wishart log-likelihood based on :cite:`malininRegressionPriorNetworks2020`.

    Used by :func:`rpn_distillation_loss` for univariate ensemble distribution
    distillation with Regression Prior Networks.

    Args:
        m (Tensor): Prior mean parameter.
        l_precision (Tensor): Precision (> 0), formerly `L`.
        kappa (Tensor): Strength parameter (> 0).
        nu (Tensor): Degrees of freedom (> 2).
        mu_k (Tensor): Sample mean from ensemble.
        sigma2_k (Tensor): Sample variance from ensemble.

    Returns:
        Tensor: Log-likelihood under the Normal-Wishart model.
    """
    # Likelihood of ensemble mean under Normal prior for mean
    log_p_mu = -0.5 * kappa * l_precision * (mu_k - m) ** 2

    # Likelihood of variance under Wishart prior on precision
    log_p_sigma = 0.5 * (nu - 1) * torch.log(l_precision) - 0.5 * nu * (sigma2_k * l_precision)

    return log_p_mu + log_p_sigma


def rpn_distillation_loss(
    rpn_params: tuple[Tensor, Tensor, Tensor, Tensor],
    mus: list[Tensor],
    variances: list[Tensor],
) -> Tensor:
    """Regression Prior Network distillation loss based on :cite:`malininRegressionPriorNetworks2020`.

    Uses ensemble distribution distillation for Regression Prior Networks (RPN).
    This loss measures how well the RPN's Normal-Wishart distribution matches the empirical ensemble
    distributions ``(mu_k, var_k)`` using :func:`normal_wishart_log_prob`.

    Args:
        rpn_params: The RPN output parameters (m, l_precision, kappa, nu).
        mus: Ensemble predicted means.
        variances: Ensemble predicted variances.

    Returns:
        Scalar loss value.

    """
    m, l_precision, kappa, nu = rpn_params  # formerly "L"

    losses: list[Tensor] = []

    for mu_k, var_k in zip(mus, variances, strict=False):
        log_prob = normal_wishart_log_prob(
            m,
            l_precision,
            kappa,
            nu,
            mu_k,
            var_k,
        )
        losses.append(-log_prob.mean())  # negative log-likelihood

    return torch.stack(losses).mean()
