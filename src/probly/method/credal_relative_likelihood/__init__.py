"""Credal relative likelihood method."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE_LIST

from ._common import (
    CredalRelativeLikelihoodPredictor,
    credal_relative_likelihood,
    relative_likelihood_thresholds,
    train_credal_relative_likelihood,
)


@train_credal_relative_likelihood.delayed_register((TORCH_MODULE_LIST, list))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "CredalRelativeLikelihoodPredictor",
    "credal_relative_likelihood",
    "relative_likelihood_thresholds",
    "train_credal_relative_likelihood",
]
