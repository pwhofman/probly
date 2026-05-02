"""Conformalized Credal Set Prediction implementation."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import (
    ConformalCredalSetPredictor,
    DirichletConformalCredalSetPredictor,
    conformal_credal_set_generator,
    conformal_dirichlet_relative_likelihood,
    conformal_inner_product,
    conformal_kullback_leibler,
    conformal_total_variation,
    conformal_wasserstein_distance,
)


@conformal_credal_set_generator.delayed_register(TORCH_MODULE)
def _(_cls: type[object]) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "ConformalCredalSetPredictor",
    "DirichletConformalCredalSetPredictor",
    "conformal_credal_set_generator",
    "conformal_dirichlet_relative_likelihood",
    "conformal_inner_product",
    "conformal_kullback_leibler",
    "conformal_total_variation",
    "conformal_wasserstein_distance",
]
