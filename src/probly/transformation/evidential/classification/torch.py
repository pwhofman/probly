"""Torch evidential classification implementation."""

from __future__ import annotations

from torch import nn

from probly.transformation.evidential.classification.common import register


def append_activation_torch(obj: nn.Module) -> nn.Sequential:
    """Register a base model that the activation function will be appended to.

    This applies a Softplus activation to ensure non-negative evidence, as required by Evidential Deep Learning.

    References:
        Based on 'Evidential Deep Learning to Quantify Classification Uncertainty' by Sensoy et al. (2018)
        See: :cite:t:`sensoyEvidentialDeep2018`

    """
    return nn.Sequential(obj, nn.Softplus())


register(nn.Module, append_activation_torch)
