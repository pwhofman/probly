"""Torch implementation of Credal Relative Likelihood."""

from __future__ import annotations

from torch import nn

from ._common import initialize_credal_relative_likelihood_model


@initialize_credal_relative_likelihood_model.register(nn.Module)
def torch_initialize_credal_relative_likelihood_model[**In, Out](base: nn.Module, reset_params: bool) -> nn.Module:
    """Initialize a torch credal relative likelihood model."""
    if reset_params:
        return base
    return base
