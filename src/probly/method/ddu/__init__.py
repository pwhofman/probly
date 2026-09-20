"""Deep Deterministic Uncertainty method."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE, TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import (
    DDUDensityDecomposition,
    DDUPredictor,
    DDURepresentation,
    create_ddu_representation,
    ddu,
    ddu_generator,
    negative_log_density,
)


@ddu_generator.delayed_register(TORCH_MODULE)
@create_ddu_representation.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@negative_log_density.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "DDUDensityDecomposition",
    "DDUPredictor",
    "DDURepresentation",
    "create_ddu_representation",
    "ddu",
]
