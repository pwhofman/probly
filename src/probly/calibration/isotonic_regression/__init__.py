"""Implementation of Isotonic Regression."""

from __future__ import annotations

from typing import Any

from probly.lazy_types import TORCH_MODULE

from .common import _isotonic_factory


# Torch
@_isotonic_factory.register(TORCH_MODULE)
def _(_base: object, _use_logits: bool) -> type[Any]:
    from . import torch  # noqa: F401, PLC0415

    return _isotonic_factory(_base, _use_logits)
