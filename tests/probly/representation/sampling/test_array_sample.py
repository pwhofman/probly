"""Tests for the ArraySample Representation."""

from __future__ import annotations

import numpy as np

from probly.representation.sample import ArraySample


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

    def test_array_function_concatenate_preserves_sample_axis(self, array_sample_2d: ArraySample[int]) -> None:
        other = ArraySample(np.arange(12, 24).reshape((3, 4)), sample_axis=1)

        result = np.concatenate((array_sample_2d, other), axis=1)

        assert isinstance(result, ArraySample)
        assert result.sample_axis == 1
        assert np.array_equal(result.array, np.concatenate((array_sample_2d.array, other.array), axis=1))

    def test_array_function_concatenate_drops_type_on_sample_axis_mismatch(
        self, array_sample_2d: ArraySample[int]
    ) -> None:
        other = ArraySample(np.arange(12, 24).reshape((3, 4)), sample_axis=0)

        result = np.concatenate((array_sample_2d, other), axis=0)

        assert isinstance(result, np.ndarray)

    def test_array_function_concatenate_axis_none_drops_type(self, array_sample_2d: ArraySample[int]) -> None:
        result = np.concatenate((array_sample_2d, array_sample_2d), axis=None)

        assert isinstance(result, np.ndarray)

    def test_array_function_concatenate_with_sample_out_returns_out(self, array_sample_2d: ArraySample[int]) -> None:
        out = ArraySample(np.empty((3, 8), dtype=array_sample_2d.dtype), sample_axis=0)

        result = np.concatenate((array_sample_2d, array_sample_2d), axis=1, out=out)

        assert result is out
        assert np.array_equal(out.array, np.concatenate((array_sample_2d.array, array_sample_2d.array), axis=1))

    def test_array_function_stack_shifts_sample_axis_when_axis_is_before(
        self, array_sample_2d: ArraySample[int]
    ) -> None:
        result = np.stack((array_sample_2d, array_sample_2d), axis=0)

        assert isinstance(result, ArraySample)
        assert result.sample_axis == array_sample_2d.sample_axis + 1
        assert np.array_equal(result.array, np.stack((array_sample_2d.array, array_sample_2d.array), axis=0))

    def test_array_function_stack_keeps_sample_axis_when_axis_is_after(self, array_sample_2d: ArraySample[int]) -> None:
        result = np.stack((array_sample_2d, array_sample_2d), axis=2)

        assert isinstance(result, ArraySample)
        assert result.sample_axis == array_sample_2d.sample_axis
        assert np.array_equal(result.array, np.stack((array_sample_2d.array, array_sample_2d.array), axis=2))

    def test_array_function_stack_drops_type_on_sample_axis_mismatch(self, array_sample_2d: ArraySample[int]) -> None:
        other = ArraySample(np.arange(12, 24).reshape((3, 4)), sample_axis=0)

        result = np.stack((array_sample_2d, other), axis=0)

        assert isinstance(result, np.ndarray)

    def test_array_function_stack_with_sample_out_returns_out(self, array_sample_2d: ArraySample[int]) -> None:
        out = ArraySample(np.empty((2, 3, 4), dtype=array_sample_2d.dtype), sample_axis=0)

        result = np.stack((array_sample_2d, array_sample_2d), axis=0, out=out)

        assert result is out
        assert np.array_equal(out.array, np.stack((array_sample_2d.array, array_sample_2d.array), axis=0))
