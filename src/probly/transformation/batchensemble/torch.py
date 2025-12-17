"""Torch Bayesian implementation."""

from __future__ import annotations

from torch import nn

from probly.layers.torch import BatchEnsembleConv2d, BatchEnsembleLinear

from .common import register


def replace_torch_batchensemble_linear(
    obj: nn.Linear,
    num_members: int,
    s_mean : float,
    s_std : float,
    r_mean : float,
    r_std : float,
) -> BatchEnsembleLinear:
    """Replace a given layer by a BatchEnsembleLinear layer."""
    return BatchEnsembleLinear(obj, num_members, s_mean, s_std, r_mean, r_std)


def replace_torch_batchensemble_conv2d(
    obj: nn.Conv2d,
    num_members: int,
    s_mean : float,
    s_std : float,
    r_mean : float,
    r_std : float,
) -> BatchEnsembleConv2d:
    """Replace a given layer by a BatchEnsembleConv2d layer."""
    return BatchEnsembleConv2d(obj, num_members, s_mean, s_std, r_mean, r_std)


register(nn.Linear, replace_torch_batchensemble_linear)
register(nn.Conv2d, replace_torch_batchensemble_conv2d)
