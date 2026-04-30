"""Module for natural posterior network implementations."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import (
    CertaintyBudget,
    NaturalPosteriorNetworkPredictor,
    natural_posterior_network,
    natural_posterior_network_generator,
)


## Torch
@natural_posterior_network_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = ["CertaintyBudget", "NaturalPosteriorNetworkPredictor", "natural_posterior_network"]
