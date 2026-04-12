"""This module contains the conformal representers for regression and classification."""

from probly.lazy_types import TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import (
    ConformalClassificationRepresenter,
    ConformalQuantileRegressionRepresenter,
    ConformalRegressionRepresenter,
    ConformalRepresenter,
    flatten_ensemble_quantile_sample,
    flatten_sample,
)


@flatten_sample.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@flatten_ensemble_quantile_sample.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "ConformalClassificationRepresenter",
    "ConformalQuantileRegressionRepresenter",
    "ConformalRegressionRepresenter",
    "ConformalRepresenter",
]
