"""Deterministic Uncertainty Quantification method."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE, TORCH_VALUE

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
@create_duq_representation.delayed_register(TORCH_VALUE)
@duq_uncertainty.delayed_register(TORCH_VALUE)
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
