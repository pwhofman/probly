"""Bayesian implementation for uncertainty quantification."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import (
    BayesianPredictor,
    bayesian,
    bayesian_traverser,
    collect_kl_divergence,
    kl_divergence_traverser,
    register,
)


## Torch
@bayesian_traverser.delayed_register(TORCH_MODULE)
@kl_divergence_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "BayesianPredictor",
    "bayesian",
    "collect_kl_divergence",
    "register",
]
