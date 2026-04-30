"""Heteroscedastic classification transformation."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import (
    HeteroscedasticClassificationPredictor,
    heteroscedastic_classification,
    heteroscedastic_classification_traverser,
)


## Torch
@heteroscedastic_classification_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "HeteroscedasticClassificationPredictor",
    "heteroscedastic_classification",
    "heteroscedastic_classification_traverser",
]
