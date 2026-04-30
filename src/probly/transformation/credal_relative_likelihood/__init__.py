"""Credal relative likelihood method."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import CredalRelativeLikelihoodPredictor, credal_relative_likelihood, credal_relative_likelihood_traverser


## Torch
@credal_relative_likelihood_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "CredalRelativeLikelihoodPredictor",
    "credal_relative_likelihood",
]
