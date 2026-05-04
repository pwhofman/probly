"""Direct Epistemic Uncertainty Prediction (DEUP) method."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE, TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import (
    DEUPDecomposition,
    DEUPPredictor,
    DEUPRepresentation,
    create_deup_representation,
    deup,
    deup_generator,
)


@deup_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@create_deup_representation.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "DEUPDecomposition",
    "DEUPPredictor",
    "DEUPRepresentation",
    "create_deup_representation",
    "deup",
    "deup_generator",
]
