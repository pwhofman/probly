"""Classes representing prediction samples."""

from __future__ import annotations

import numpy as np

from lazy_dispatch.singledispatch import lazydispatch
from probly.lazy_types import JAX_ARRAY, TORCH_TENSOR
from probly.representation.sampling.array_sample import ArraySample
from probly.representation.sampling.common_sample import ListSample, Sample, SampleFactory

create_sample = lazydispatch[SampleFactory, Sample](
    ListSample.from_iterable,
    dispatch_on=lambda s, sample_axis=1: s[0],  # noqa: ARG005 sample_axis is unused
)


create_sample.register(np.number | np.ndarray | float | int, ArraySample.from_iterable)


@create_sample.delayed_register(TORCH_TENSOR)
def _(_: type) -> None:
    from . import torch_sample as torch_sample  # noqa: PLC0414, PLC0415


@create_sample.delayed_register(JAX_ARRAY)
def _(_: type) -> None:
    from . import jax_sample as jax_sample  # noqa: PLC0414, PLC0415


__all__ = ["ArraySample", "ListSample", "Sample", "SampleFactory", "create_sample"]
