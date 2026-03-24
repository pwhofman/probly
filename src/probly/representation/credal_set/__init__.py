"""Credal set representations."""

from __future__ import annotations

from probly.lazy_types import TORCH_TENSOR

from .array import ArrayCategoricalCredalSet, ArrayDiscreteCredalSet, create_probability_intervals
from .common import CategoricalCredalSet, CredalSet, DiscreteCredalSet


@create_probability_intervals.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "ArrayCategoricalCredalSet",
    "ArrayDiscreteCredalSet",
    "CategoricalCredalSet",
    "CredalSet",
    "DiscreteCredalSet",
    "create_probability_intervals",
]
