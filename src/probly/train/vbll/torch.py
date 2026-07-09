"""Collection of torch VBLL training losses.

These functions implement the negative ELBO objectives used to fit the variational
Bayesian last layers of :cite:`harrisonVariationalBayesian2024`. Each takes the layer
together with the backbone features feeding it (captured with a forward pre-hook),
the integer targets, and the weight on the KL/regularization terms.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch.nn import functional as F

from probly.layers.torch import (
    GVBLLLayer,
    HetVBLLLayer,
    TVBLLLayer,
    VBLLLayer,
    VBLLParameterization,
    _vbll_logit_variance,
)

from ._common import vbll_loss

__all__ = [
    "disc_vbll_loss",
    "g_vbll_loss",
    "het_vbll_loss",
    "t_vbll_loss",
    "vbll_loss",
]


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


@vbll_loss.register(VBLLLayer)
def disc_vbll_loss(
    layer: VBLLLayer,
    features: torch.Tensor,
    targets: torch.Tensor,
    regularization_weight: float,
) -> torch.Tensor:
    """Negative discriminative ELBO of a :class:`~probly.layers.torch.VBLLLayer` using the double-Jensen bound.

    Implements the discriminative classification objective of
    :cite:`harrisonVariationalBayesian2024`: the closed-form double-Jensen lower
    bound on the expected log-likelihood, regularized by the weight-posterior
    :attr:`~probly.layers.torch.VBLLLayer.kl_divergence`. Both ingredients of the
    bound - the logit mean and the logit variance ``phi^T S_k phi + sigma_k^2`` -
    are exactly the ``(mean, var)`` returned by the layer's forward pass.

    Args:
        layer: The variational Bayesian last layer to fit.
        features: Backbone features feeding the layer, shape ``(batch, in_features)``.
        targets: Integer class labels, shape ``(batch,)``.
        regularization_weight: Weight on the KL term (typically ``1 / dataset_size``).

    Returns:
        A scalar tensor with the negative ELBO to minimize.
    """
    mean, var = layer(features)
    index = torch.arange(features.shape[0])
    true_logit = mean[index, targets]
    log_normalizer = torch.logsumexp(mean + 0.5 * var, dim=-1)
    expected_log_likelihood = (true_logit - log_normalizer).mean()
    return -expected_log_likelihood + regularization_weight * layer.kl_divergence


@vbll_loss.register(GVBLLLayer)
def g_vbll_loss(
    layer: GVBLLLayer,
    features: torch.Tensor,
    targets: torch.Tensor,
    regularization_weight: float,
) -> torch.Tensor:
    """Negative generative ELBO (the Jensen bound) of a :class:`~probly.layers.torch.GVBLLLayer`.

    Implements the discriminative-free generative training objective of
    :cite:`harrisonVariationalBayesian2024`: the Jensen lower bound on the expected
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

    wishart_term = layer.dof * (-noise_log_var.sum()) - 0.5 * layer.wishart_scale * torch.exp(-noise_log_var).sum()

    total_elbo = jensen_bound.mean() + regularization_weight * (wishart_term - layer.kl_divergence)
    return -total_elbo


@vbll_loss.register(TVBLLLayer)
def t_vbll_loss(
    layer: TVBLLLayer,
    features: torch.Tensor,
    targets: torch.Tensor,
    regularization_weight: float,
) -> torch.Tensor:
    """Negative ELBO of a :class:`~probly.layers.torch.TVBLLLayer` using the reduced Knowles-Minka bound.

    Implements the Student-t discriminative objective of
    :cite:`harrisonVariationalBayesian2024`, combining the reduced Knowles-Minka
    softmax bound with the Gamma noise-precision KL and the weight-posterior KL.

    Args:
        layer: The Student-t variational Bayesian last layer to fit.
        features: Backbone features feeding the layer, shape ``(batch, in_features)``.
        targets: Integer class labels, shape ``(batch,)``.
        regularization_weight: Weight on the regularization terms (typically ``1 / dataset_size``).

    Returns:
        A scalar tensor with the negative ELBO to minimize.
    """
    offdiag = getattr(layer, "W_offdiag", None)
    mean = F.linear(features, layer.W_mean)
    weight_variance = _vbll_logit_variance(features, layer.W_logdiag, offdiag, layer.parameterization)
    cov = weight_variance + 1.0
    index = torch.arange(features.shape[0])
    linear_term = mean[index, targets]
    lse_term = torch.logsumexp(mean + layer.alpha * cov, dim=-1)

    expected_cov = torch.exp(layer.noise_log_rate - layer.noise_log_dof + 1.0)
    expected_prec = torch.exp(layer.noise_log_dof - layer.noise_log_rate)
    cov_term = cov * (expected_cov / 4.0 + expected_prec * layer.alpha**2 - layer.alpha)
    bound = linear_term - lse_term - 0.5 * cov_term.sum(dim=-1)

    noise = torch.distributions.Gamma(torch.exp(layer.noise_log_dof), torch.exp(layer.noise_log_rate))
    prior = torch.distributions.Gamma(
        torch.full_like(layer.noise_log_dof, layer.prior_dof),
        torch.full_like(layer.noise_log_rate, layer.prior_rate),
    )
    gamma_kl = torch.distributions.kl_divergence(noise, prior).sum(dim=-1)
    cov_factor = torch.exp(layer.noise_log_dof - layer.noise_log_rate)
    weight_kl = _gaussian_weight_kl(
        layer.W_mean, layer.W_logdiag, offdiag, layer.parameterization, layer.prior_scale, cov_factor
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
    """Negative ELBO of a :class:`~probly.layers.torch.HetVBLLLayer` using the reduced Knowles-Minka bound.

    Implements the heteroscedastic discriminative objective of
    :cite:`harrisonVariationalBayesian2024`, combining the reduced Knowles-Minka
    softmax bound with the input-dependent noise KL and the weight-posterior KL.

    Args:
        layer: The heteroscedastic variational Bayesian last layer to fit.
        features: Backbone features feeding the layer, shape ``(batch, in_features)``.
        targets: Integer class labels, shape ``(batch,)``.
        regularization_weight: Weight on the regularization terms (typically ``1 / dataset_size``).

    Returns:
        A scalar tensor with the negative ELBO to minimize.
    """
    w_offdiag = getattr(layer, "W_offdiag", None)
    m_offdiag = getattr(layer, "M_offdiag", None)
    mean = F.linear(features, layer.W_mean)
    weight_variance = _vbll_logit_variance(features, layer.W_logdiag, w_offdiag, layer.parameterization)
    cov = weight_variance + 1.0
    index = torch.arange(features.shape[0])
    linear_term = mean[index, targets]
    lse_term = torch.logsumexp(mean + layer.alpha * cov, dim=-1)

    log_noise_mean = F.linear(features, layer.M_mean)
    log_noise_var = _vbll_logit_variance(features, layer.M_logdiag, m_offdiag, layer.parameterization)
    expected_cov = torch.exp(log_noise_mean + 0.5 * log_noise_var)
    expected_prec = torch.exp(-log_noise_mean + 0.5 * log_noise_var)
    cov_term = cov * (expected_cov / 4.0 + expected_prec * layer.alpha**2 - layer.alpha)
    bound = linear_term - lse_term - 0.5 * cov_term.sum(dim=-1)

    weight_kl = _gaussian_weight_kl(
        layer.W_mean, layer.W_logdiag, w_offdiag, layer.parameterization, layer.prior_scale, expected_prec
    ).mean()
    noise_kl = _gaussian_weight_kl(
        layer.M_mean, layer.M_logdiag, m_offdiag, layer.parameterization, layer.noise_prior_scale
    )

    total_elbo = bound.mean() - regularization_weight * (weight_kl + noise_kl)
    return -total_elbo
