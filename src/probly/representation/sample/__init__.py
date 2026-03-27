"""Classes representing prediction samples."""

from __future__ import annotations

from probly.lazy_types import JAX_ARRAY, TORCH_TENSOR

from ._common import ListSample, Sample, SampleAxis, SampleFactory, SampleParams, create_sample
from .array import ArraySample


@create_sample.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@create_sample.delayed_register(JAX_ARRAY)
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415


__all__ = [
    "ArraySample",
    "ListSample",
    "Sample",
    "SampleAxis",
    "SampleFactory",
    "SampleParams",
    "create_sample",
]
