"""Deterministic Uncertainty Quantification (DUQ) representations."""

from __future__ import annotations

from probly.lazy_types import TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import DUQRepresentation, create_duq_representation


@create_duq_representation.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "DUQRepresentation",
    "create_duq_representation",
]
