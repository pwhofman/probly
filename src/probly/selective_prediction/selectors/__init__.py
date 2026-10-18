"""Selectors: rules on the uncertainty criterion that accept or reject predictions, see :func:`selector`."""

from __future__ import annotations

from probly.lazy_types import TORCH_TENSOR

from ._common import Selector, SelectorName, _to_float64_numpy, selector, selector_registry


@_to_float64_numpy.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "Selector",
    "SelectorName",
    "selector",
    "selector_registry",
]
