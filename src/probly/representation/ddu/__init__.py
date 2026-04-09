"""Deep Deterministic Uncertainty representations."""

from __future__ import annotations

from probly.lazy_types import TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import DDURepresentation, create_ddu_representation


@create_ddu_representation.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "DDURepresentation",
    "create_ddu_representation",
]
