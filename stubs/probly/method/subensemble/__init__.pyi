"""Subensemble implementation for uncertainty quantification."""

from __future__ import annotations

from probly.lazy_types import FLAX_MODULE, TORCH_MODULE

from ._common import SubensemblePredictor, subensemble, subensemble_generator


## Torch
@subensemble_generator.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    ...


## Flax
@subensemble_generator.delayed_register(FLAX_MODULE)
def _(_: type) -> None:
    ...


__all__ = [
    "SubensemblePredictor",
    "subensemble",
    "subensemble_generator",
]
