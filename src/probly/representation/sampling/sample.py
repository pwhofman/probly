"""Classes representing prediction samples."""

from __future__ import annotations

from probly.lazy_types import JAX_ARRAY, TORCH_TENSOR
from probly.representation.sampling.array_sample import ArraySample
from probly.representation.sampling.common_sample import ListSample, Sample, SampleFactory, create_sample


@create_sample.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch_sample as torch_sample  # noqa: PLC0414, PLC0415


@create_sample.delayed_register(JAX_ARRAY)
def _(_: type) -> None:
    from . import jax_sample as jax_sample  # noqa: PLC0414, PLC0415


__all__ = ["ArraySample", "ListSample", "Sample", "SampleFactory", "create_sample"]
