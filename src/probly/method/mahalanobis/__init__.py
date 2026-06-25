"""Mahalanobis out-of-distribution detection method."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE, TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import (
    MahalanobisDecomposition,
    MahalanobisPredictor,
    MahalanobisRepresentation,
    combine_layer_scores,
    create_mahalanobis_representation,
    mahalanobis,
    mahalanobis_generator,
)


@mahalanobis_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@create_mahalanobis_representation.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@combine_layer_scores.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "MahalanobisDecomposition",
    "MahalanobisPredictor",
    "MahalanobisRepresentation",
    "create_mahalanobis_representation",
    "mahalanobis",
]
