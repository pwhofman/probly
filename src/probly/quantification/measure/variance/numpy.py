"""Numpy implementations of regression variance measures."""

from __future__ import annotations

import numpy as np

from probly.representation.sample.numpy import NumpySample

from ._common import LogBase, expected_conditional_variance


@expected_conditional_variance.register(NumpySample)
def numpy_sample_expected_conditional_variance(
    sample: NumpySample,
    base: LogBase = None,  # noqa: ARG001
) -> np.ndarray:
    """Compute the expected conditional variance of a raw numpy array sample (zero for point predictions)."""
    return np.zeros_like(np.mean(sample.array, axis=sample.sample_axis))
