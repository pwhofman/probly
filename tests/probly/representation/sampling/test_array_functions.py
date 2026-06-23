"""Tests mirroring src/probly/representation/sample/array_functions.py."""

from __future__ import annotations

import numpy as np  # noqa: F401
import pytest

from probly.representation.sample.array import ArraySample  # noqa: F401
from probly.representation.sample.array_functions import (
    array_sample_internals,  # noqa: F401
    track_sample_axis_after_reduction,
)


def _torch_modules():
    """Skip the calling test if torch is unavailable; otherwise return the module."""
    pytest.importorskip("torch")
    import torch as _torch  # noqa: PLC0415

    return _torch


def _jax_modules():
    """Skip the calling test if jax is unavailable; otherwise return (jax, jnp)."""
    pytest.importorskip("jax")
    import jax as _jax  # noqa: PLC0415
    import jax.numpy as _jnp  # noqa: PLC0415

    return _jax, _jnp


class TestTrackSampleAxisAfterReduction:
    """Pure helper: compute new sample axis after a reduction."""

    def test_axis_none_returns_none(self) -> None:
        assert track_sample_axis_after_reduction(0, 3, axis=None, keepdims=False) is None

    def test_keepdims_preserves_axis(self) -> None:
        assert track_sample_axis_after_reduction(2, 4, axis=1, keepdims=True) == 2

    def test_reduction_along_sample_axis_returns_none(self) -> None:
        assert track_sample_axis_after_reduction(1, 3, axis=1, keepdims=False) is None

    def test_reduction_before_sample_axis_shifts(self) -> None:
        # Sample at axis 2; reduce axis 0; new sample axis is 1.
        assert track_sample_axis_after_reduction(2, 3, axis=0, keepdims=False) == 1

    def test_reduction_after_sample_axis_unchanged(self) -> None:
        assert track_sample_axis_after_reduction(0, 3, axis=2, keepdims=False) == 0

    def test_reduction_negative_axis_normalised(self) -> None:
        # Negative axis -1 (== axis 2) for a 3-D array; sample at 0 -> unchanged.
        assert track_sample_axis_after_reduction(0, 3, axis=-1, keepdims=False) == 0

    def test_tuple_axis(self) -> None:
        # Sample at axis 2; reduce axes (0, 1); new sample axis is 0.
        assert track_sample_axis_after_reduction(2, 3, axis=(0, 1), keepdims=False) == 0
