"""Tests for the ArraySample Representation."""

from __future__ import annotations

import numpy as np

from probly.representation.sampling.sample import ArraySample


class TestArraySample:
    def test_sample_internal_array(self, array_sample_2d: ArraySample[int]) -> None:
        assert isinstance(array_sample_2d.array, np.ndarray)

    def test_sample_length(self, array_sample_2d: ArraySample[int]) -> None:
        assert len(array_sample_2d) == len(array_sample_2d.array)

    def test_sample_ndim(self, array_sample_2d: ArraySample[int]) -> None:
        assert array_sample_2d.ndim == 2

    def test_sample_shape(self, array_sample_2d: ArraySample[int]) -> None:
        assert array_sample_2d.shape == array_sample_2d.array.shape

    def test_sample_move_axis(self, array_sample_2d: ArraySample[int]) -> None:
        moved_sample = array_sample_2d.move_sample_axis(0)
        assert isinstance(moved_sample, ArraySample)
        assert moved_sample.sample_axis == 0
        assert array_sample_2d.shape[array_sample_2d.sample_axis] == moved_sample.shape[moved_sample.sample_axis]

    def test_sample_concat(self, array_sample_2d: ArraySample[int]) -> None:
        res = array_sample_2d.concat(array_sample_2d.move_sample_axis(0))
        assert isinstance(res, ArraySample)
        assert res.sample_axis == array_sample_2d.sample_axis
        assert res.sample_size == 2 * array_sample_2d.sample_size

    def test_sample_slicing(self, array_sample_2d: ArraySample[int]) -> None:
        indexed_sample = array_sample_2d[:, :3]

        assert isinstance(indexed_sample, ArraySample)
        assert indexed_sample.sample_axis == 1
        assert indexed_sample.shape == (3, 3)

    def test_sample_selection(self, array_sample_2d: ArraySample[int]) -> None:
        indexed_sample = array_sample_2d[:, 3]

        assert isinstance(indexed_sample, np.ndarray)
        assert indexed_sample.shape == (3,)

    def test_ufunc_call(self, array_sample_2d: ArraySample[int]) -> None:
        result = array_sample_2d + 5

        assert isinstance(result, ArraySample)
        assert result.sample_axis == array_sample_2d.sample_axis
        assert np.array_equal(result.array, array_sample_2d.array + 5)

    def test_ufunc_reduce_along_non_sample(self, array_sample_2d: ArraySample[int]) -> None:
        result = np.add.reduce(array_sample_2d, axis=0)

        assert isinstance(result, ArraySample)
        assert result.sample_axis == array_sample_2d.sample_axis - 1
        expected_array = np.add.reduce(array_sample_2d.array, axis=0)
        assert np.array_equal(result.array, expected_array)

    def test_ufunc_reduce_along_sample(self, array_sample_2d: ArraySample[int]) -> None:
        result = np.add.reduce(array_sample_2d, axis=-1)

        assert isinstance(result, np.ndarray)
        expected_array = np.add.reduce(array_sample_2d.array, axis=-1)
        assert np.array_equal(result, expected_array)
