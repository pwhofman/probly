"""Credal set representations."""

from __future__ import annotations

from probly.lazy_types import TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import (
    CategoricalCredalSet,
    CredalSet,
    CredalSetType,
    DiscreteCredalSet,
    ProbabilityIntervalsCredalSet,
    create_convex_credal_set,
    create_probability_intervals,
    create_probability_intervals_from_bounds,
    create_probability_intervals_from_lower_upper_array,
)
from .array import ArrayCategoricalCredalSet, ArrayDiscreteCredalSet


@create_probability_intervals.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@create_probability_intervals_from_lower_upper_array.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@create_convex_credal_set.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
@create_probability_intervals_from_bounds.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "ArrayCategoricalCredalSet",
    "ArrayDiscreteCredalSet",
    "CategoricalCredalSet",
    "CredalSet",
    "CredalSetType",
    "DiscreteCredalSet",
    "ProbabilityIntervalsCredalSet",
    "create_convex_credal_set",
    "create_probability_intervals",
    "create_probability_intervals_from_bounds",
    "create_probability_intervals_from_lower_upper_array",
]
