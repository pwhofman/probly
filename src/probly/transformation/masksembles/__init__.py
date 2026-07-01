"""Masksembles implementation for uncertainty quantification."""

from __future__ import annotations

from probly.lazy_types import TORCH_MODULE

from ._common import MasksemblesPredictor, masksembles, masksembles_traverser, register


@masksembles_traverser.delayed_register(TORCH_MODULE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = ["MasksemblesPredictor", "masksembles", "masksembles_traverser", "register"]
