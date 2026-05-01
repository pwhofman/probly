"""Classes representing prediction samples."""

from __future__ import annotations

from probly.lazy_types import JAX_ARRAY, JAX_ARRAY_LIKE, TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import ListSample, Sample, SampleAxis, SampleFactory, SampleParams, create_sample
from .array import ArraySample


@create_sample.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@create_sample.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
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
