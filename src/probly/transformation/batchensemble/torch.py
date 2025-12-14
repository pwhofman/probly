"""Torch Bayesian implementation."""

from __future__ import annotations

from torch import nn

from probly.layers.torch import BatchEnsembleConv2d, BatchEnsembleLinear

from .common import register


def replace_torch_batchensemble_linear(
    obj: nn.Linear,
    use_base_weights: bool,
    posterior_std: float,
    prior_mean: float,
    prior_std: float,
) -> BatchEnsembleLinear:
    """Replace a given layer by a BatchEnsembleLinear layer."""
    return BatchEnsembleLinear(obj, use_base_weights, posterior_std, prior_mean, prior_std)


def replace_torch_batchensemble_conv2d(
    obj: nn.Conv2d,
    use_base_weights: bool,
    posterior_std: float,
    prior_mean: float,
    prior_std: float,
) -> BatchEnsembleConv2d:
    """Replace a given layer by a BatchEnsembleConv2d layer."""
    return BatchEnsembleConv2d(obj, use_base_weights, posterior_std, prior_mean, prior_std)


register(nn.Linear, replace_torch_batchensemble_linear)
register(nn.Conv2d, replace_torch_batchensemble_conv2d)
