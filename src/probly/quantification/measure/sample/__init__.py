"""Uncertainty measures for samples."""

from probly.lazy_types import TORCH_SAMPLE

from ._common import mean_squared_distance_to_scaled_one_hot, measure_sample_variance
from .array import array_mean_squared_distance_to_scaled_one_hot


@mean_squared_distance_to_scaled_one_hot.delayed_register(TORCH_SAMPLE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = [
    "array_mean_squared_distance_to_scaled_one_hot",
    "mean_squared_distance_to_scaled_one_hot",
    "measure_sample_variance",
]
