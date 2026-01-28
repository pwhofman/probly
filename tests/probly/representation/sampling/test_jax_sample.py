"""Tests for the JaxArraySample Representation."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")
from jax import numpy as jnp

from probly.representation.sampling.jax_sample import JaxArraySample


class TestJaxArraySample:
    def test_sample_internal_array(self, jax_array_sample_2d: JaxArraySample) -> None:
        assert isinstance(jax_array_sample_2d.array, jnp.ndarray)

    def test_sample_length(self, jax_array_sample_2d: JaxArraySample) -> None:
        assert len(jax_array_sample_2d) == len(jax_array_sample_2d.array)

    def test_sample_ndim(self, jax_array_sample_2d: JaxArraySample) -> None:
        assert jax_array_sample_2d.ndim == 2

    def test_sample_shape(self, jax_array_sample_2d: JaxArraySample) -> None:
        assert jax_array_sample_2d.shape == jax_array_sample_2d.array.shape

    def test_sample_move_axis(self, jax_array_sample_2d: JaxArraySample) -> None:
        moved_sample = jax_array_sample_2d.move_sample_axis(0)
        assert isinstance(moved_sample, JaxArraySample)
        assert moved_sample.sample_axis == 0
        assert (
            jax_array_sample_2d.shape[jax_array_sample_2d.sample_axis] == moved_sample.shape[moved_sample.sample_axis]
        )

    def test_sample_concat(self, jax_array_sample_2d: JaxArraySample) -> None:
        res = jax_array_sample_2d.concat(jax_array_sample_2d.move_sample_axis(0))
        assert isinstance(res, JaxArraySample)
        assert res.sample_axis == jax_array_sample_2d.sample_axis
        assert res.sample_size == 2 * jax_array_sample_2d.sample_size
