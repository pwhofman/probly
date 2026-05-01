"""Deterministic Uncertainty Quantification method."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE, TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import (
    DUQDecomposition,
    DUQPredictor,
    DUQRepresentation,
    create_duq_representation,
    duq,
    duq_generator,
    duq_uncertainty,
)


@duq_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@create_duq_representation.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@duq_uncertainty.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "DUQDecomposition",
    "DUQPredictor",
    "DUQRepresentation",
    "create_duq_representation",
    "duq",
    "duq_uncertainty",
]
