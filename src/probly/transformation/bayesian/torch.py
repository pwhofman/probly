"""Torch Bayesian implementation."""

from __future__ import annotations

from torch import nn

from probly.layers.torch import BayesConv2d, BayesLinear

from .common import register


def replace_torch_bayesian_linear(
    obj: nn.Linear,
    use_base_weights: bool,
    posterior_std: float,
    prior_mean: float,
    prior_std: float,
) -> BayesLinear:
    """Replace a given layer by a BayesLinear layer based on :cite:`blundellWeightUncertainty2015`."""
    return BayesLinear(obj, use_base_weights, posterior_std, prior_mean, prior_std)


def replace_torch_bayesian_conv2d(
    obj: nn.Conv2d,
    use_base_weights: bool,
    posterior_std: float,
    prior_mean: float,
    prior_std: float,
) -> BayesConv2d:
    """Replace a given layer by a BayesConv2d layer based on :cite:`blundellWeightUncertainty2015`."""
    return BayesConv2d(obj, use_base_weights, posterior_std, prior_mean, prior_std)


register(nn.Linear, replace_torch_bayesian_linear)
register(nn.Conv2d, replace_torch_bayesian_conv2d)
