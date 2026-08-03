"""SWA-Gaussian (SWAG) method for uncertainty quantification."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import (
    SWAGPredictor,
    collect_swag,
    swag,
    swag_from_snapshots,
    swag_generator,
    swag_snapshot_generator,
)


## Torch
@swag_generator.delayed_register(TORCH_MODULE)
@swag_snapshot_generator.delayed_register(TORCH_MODULE)
@collect_swag.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "SWAGPredictor",
    "collect_swag",
    "swag",
    "swag_from_snapshots",
    "swag_generator",
    "swag_snapshot_generator",
]
