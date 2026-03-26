"""Classes representing prediction samples."""

from __future__ import annotations

from probly.lazy_types import JAX_ARRAY, TORCH_TENSOR

from .array import ArraySample
from .common import ListSample, Sample, SampleFactory, create_sample


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
    "SampleFactory",
    "create_sample",
]
