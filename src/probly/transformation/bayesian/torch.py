"""Torch Bayesian implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from probly.layers.torch import BayesConv2d, BayesLinear

from ._common import KL_DIVERGENCE, kl_divergence_traverser, register

if TYPE_CHECKING:
    from pytraverse import State, TraverserResult


def torch_replace_bayesian_linear(
    obj: nn.Linear,
    use_base_weights: bool,
    posterior_std: float,
    prior_mean: float,
    prior_std: float,
) -> BayesLinear:
    """Replace a given layer by a BayesLinear layer based on :cite:`blundellWeightUncertainty2015`."""
    return BayesLinear(obj, use_base_weights, posterior_std, prior_mean, prior_std)


def torch_replace_bayesian_conv2d(
    obj: nn.Conv2d,
    use_base_weights: bool,
    posterior_std: float,
    prior_mean: float,
    prior_std: float,
) -> BayesConv2d:
    """Replace a given layer by a BayesConv2d layer based on :cite:`blundellWeightUncertainty2015`."""
    return BayesConv2d(obj, use_base_weights, posterior_std, prior_mean, prior_std)


register(nn.Linear, torch_replace_bayesian_linear)
register(nn.Conv2d, torch_replace_bayesian_conv2d)


@kl_divergence_traverser.register(BayesLinear | BayesConv2d)
def _torch_layer_kl_divergence(
    obj: BayesLinear | BayesConv2d,
    state: State,
) -> TraverserResult[BayesLinear | BayesConv2d]:
    """Traverser to compute the KL divergence of a Bayesian layer based on :cite:`galDropoutBayesian2016`."""
    state[KL_DIVERGENCE] += obj.kl_divergence
    return obj, state
