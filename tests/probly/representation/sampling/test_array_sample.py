"""Tests for the ArraySample Representation."""

from __future__ import annotations

import numpy as np
import pytest

from probly.representation.distribution.array_categorical import ArrayCategoricalDistribution
from probly.representation.sample import ArraySample


def assert_weights_equal(sample: ArraySample, expected: object) -> None:
    assert sample.weights is not None
    assert np.array_equal(sample.weights, np.asarray(expected))


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

    def test_from_iterable_preserves_weights(self) -> None:
        weights = np.array([0.1, 0.2, 0.3])

        sample = ArraySample.from_iterable(np.arange(6).reshape((3, 2)), sample_axis=0, weights=weights)

        assert_weights_equal(sample, weights)

    def test_constructor_rejects_wrong_weight_shape(self) -> None:
        with pytest.raises(ValueError, match="weights must have shape"):
            ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=np.array([0.1, 0.2, 0.3]))

    def test_from_sample_preserves_weights(self) -> None:
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        sample = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=weights)

        converted = ArraySample.from_sample(sample, sample_axis=0)

        assert converted.sample_axis == 0
        assert_weights_equal(converted, weights)

    def test_copy_preserves_weights(self) -> None:
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        sample = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=weights)

        copied = sample.copy()

        assert copied is not sample
        assert_weights_equal(copied, weights)

    def test_sample_move_axis_preserves_weights(self) -> None:
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        sample = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=weights)

        moved_sample = sample.move_sample_axis(0)

        assert moved_sample.sample_axis == 0
        assert_weights_equal(moved_sample, weights)

    def test_sample_concat_combines_weights(self) -> None:
        left = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=np.array([0.1, 0.2, 0.3, 0.4]))
        right = ArraySample(np.arange(12, 24).reshape((4, 3)), sample_axis=0, weights=np.array([0.5, 0.6, 0.7, 0.8]))

        result = left.concat(right)

        assert result.sample_axis == 1
        assert_weights_equal(result, np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))

    def test_sample_concat_fills_missing_weights_with_ones(self) -> None:
        left = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1)
        right = ArraySample(np.arange(12, 24).reshape((3, 4)), sample_axis=1, weights=np.array([0.5, 0.6, 0.7, 0.8]))

        result = left.concat(right)

        assert_weights_equal(result, np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.6, 0.7, 0.8]))

    def test_sample_mean_uses_weights(self) -> None:
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        sample = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=weights)

        result = sample.sample_mean()

        assert np.allclose(result, np.average(sample.array, axis=1, weights=weights))

    def test_sample_mean_of_categorical_distribution_preserves_distribution_type(self) -> None:
        probabilities = np.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
        sample = ArraySample(ArrayCategoricalDistribution(probabilities), sample_axis=0)

        result = sample.sample_mean()

        assert isinstance(result, ArrayCategoricalDistribution)
        assert result.shape == (3,)
        np.testing.assert_allclose(result.unnormalized_probabilities, np.mean(probabilities, axis=0))

    def test_weighted_sample_mean_of_categorical_distribution_uses_weights(self) -> None:
        probabilities = np.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
        weights = np.array([0.25, 0.75])
        sample = ArraySample(ArrayCategoricalDistribution(probabilities), sample_axis=0, weights=weights)

        result = sample.sample_mean()

        assert isinstance(result, ArrayCategoricalDistribution)
        assert result.shape == (3,)
        np.testing.assert_allclose(
            result.unnormalized_probabilities,
            np.average(probabilities, axis=0, weights=weights),
        )

    def test_sample_var_and_std_use_weights(self) -> None:
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        sample = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=weights)
        average = np.average(sample.array, axis=1, weights=weights, keepdims=True)
        expected_var = np.average((sample.array - average) ** 2, axis=1, weights=weights)

        assert np.allclose(sample.sample_var(), expected_var)
        assert np.allclose(sample.sample_std(), np.sqrt(expected_var))

    def test_weighted_sample_var_rejects_ddof(self) -> None:
        sample = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=np.ones(4))

        with pytest.raises(ValueError, match="ddof"):
            sample.sample_var(ddof=1)

    def test_sample_slicing(self, array_sample_2d: ArraySample[int]) -> None:
        indexed_sample = array_sample_2d[:, :3]

        assert isinstance(indexed_sample, ArraySample)
        assert indexed_sample.sample_axis == 1
        assert indexed_sample.shape == (3, 3)

    def test_sample_selection(self, array_sample_2d: ArraySample[int]) -> None:
        indexed_sample = array_sample_2d[:, 3]

        assert isinstance(indexed_sample, np.ndarray)
        assert indexed_sample.shape == (3,)

    def test_weighted_non_sample_slice_preserves_weights(self) -> None:
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        sample = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=weights)

        indexed_sample = sample[:2, :]

        assert isinstance(indexed_sample, ArraySample)
        assert indexed_sample.sample_axis == 1
        assert_weights_equal(indexed_sample, weights)

    def test_weighted_sample_axis_slice_indexes_weights(self) -> None:
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        sample = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=weights)

        indexed_sample = sample[:, 1:3]

        assert isinstance(indexed_sample, ArraySample)
        assert indexed_sample.sample_axis == 1
        assert_weights_equal(indexed_sample, np.array([0.2, 0.3]))

    def test_weighted_integer_before_sample_axis_preserves_weights(self) -> None:
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        sample = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=weights)

        indexed_sample = sample[0, :]

        assert isinstance(indexed_sample, ArraySample)
        assert indexed_sample.sample_axis == 0
        assert_weights_equal(indexed_sample, weights)

    def test_weighted_integer_on_sample_axis_returns_array(self) -> None:
        sample = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=np.array([0.1, 0.2, 0.3, 0.4]))

        indexed_sample = sample[:, 2]

        assert isinstance(indexed_sample, np.ndarray)

    def test_weighted_1d_integer_index_on_sample_axis_indexes_weights(self) -> None:
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        sample = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=weights)

        indexed_sample = sample[:, np.array([3, 1])]

        assert isinstance(indexed_sample, ArraySample)
        assert_weights_equal(indexed_sample, np.array([0.4, 0.2]))

    def test_weighted_1d_boolean_index_on_sample_axis_indexes_weights(self) -> None:
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        sample = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=weights)

        indexed_sample = sample[:, np.array([True, False, True, False])]

        assert isinstance(indexed_sample, ArraySample)
        assert_weights_equal(indexed_sample, np.array([0.1, 0.3]))

    def test_weighted_newaxis_preserves_weights(self) -> None:
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        sample = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=weights)

        indexed_sample = sample[None, :, :]

        assert isinstance(indexed_sample, ArraySample)
        assert indexed_sample.sample_axis == 2
        assert_weights_equal(indexed_sample, weights)

    def test_weighted_ellipsis_sample_axis_slice_indexes_weights(self) -> None:
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        sample = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=weights)

        indexed_sample = sample[..., 1:3]

        assert isinstance(indexed_sample, ArraySample)
        assert_weights_equal(indexed_sample, np.array([0.2, 0.3]))

    def test_weighted_multidimensional_integer_index_on_sample_axis_returns_array(self) -> None:
        sample = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=np.array([0.1, 0.2, 0.3, 0.4]))

        indexed_sample = sample[:, np.array([[0, 1]])]

        assert isinstance(indexed_sample, np.ndarray)
        assert indexed_sample.shape == (3, 1, 2)

    def test_weighted_multidimensional_boolean_index_touching_sample_axis_raises(self) -> None:
        sample = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=0, weights=np.array([0.1, 0.2, 0.3]))

        with pytest.raises(IndexError, match="Weighted samples"):
            sample[np.array([[True, False, True, False], [False, True, False, True], [True, False, False, True]])]

    def test_unweighted_complex_indexing_still_works(self) -> None:
        sample = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1)

        indexed_sample = sample[:, np.array([[0, 1]])]

        assert isinstance(indexed_sample, np.ndarray)

    def test_ufunc_call(self, array_sample_2d: ArraySample[int]) -> None:
        result = array_sample_2d + 5

        assert isinstance(result, ArraySample)
        assert result.sample_axis == array_sample_2d.sample_axis
        assert np.array_equal(result.array, array_sample_2d.array + 5)

    def test_ufunc_call_preserves_weights(self) -> None:
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        sample = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=weights)

        result = sample + 5

        assert isinstance(result, ArraySample)
        assert_weights_equal(result, weights)

    def test_ufunc_reduce_along_non_sample_preserves_weights(self) -> None:
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        sample = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=weights)

        result = np.add.reduce(sample, axis=0)

        assert isinstance(result, ArraySample)
        assert_weights_equal(result, weights)

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

    def test_array_function_copy_preserves_weights(self) -> None:
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        sample = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=weights)

        result = np.copy(sample, subok=True)

        assert isinstance(result, ArraySample)
        assert_weights_equal(result, weights)

    def test_array_function_transpose_preserves_weights(self) -> None:
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        sample = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=weights)

        result = np.transpose(sample)

        assert isinstance(result, ArraySample)
        assert result.sample_axis == 0
        assert_weights_equal(result, weights)

    def test_array_function_expand_dims_preserves_weights(self) -> None:
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        sample = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=weights)

        result = np.expand_dims(sample, axis=0)

        assert isinstance(result, ArraySample)
        assert result.sample_axis == 2
        assert_weights_equal(result, weights)

    def test_array_function_squeeze_non_sample_axis_preserves_weights(self) -> None:
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        sample = ArraySample(np.arange(4).reshape((1, 4)), sample_axis=1, weights=weights)

        result = np.squeeze(sample, axis=0)

        assert isinstance(result, ArraySample)
        assert result.sample_axis == 0
        assert_weights_equal(result, weights)

    def test_array_function_squeeze_sample_axis_drops_type(self) -> None:
        sample = ArraySample(np.arange(3).reshape((3, 1)), sample_axis=1, weights=np.array([0.1]))

        result = np.squeeze(sample, axis=1)

        assert isinstance(result, np.ndarray)

    def test_array_function_concatenate_combines_weights(self) -> None:
        left = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=np.array([0.1, 0.2, 0.3, 0.4]))
        right = ArraySample(np.arange(12, 24).reshape((3, 4)), sample_axis=1, weights=np.array([0.5, 0.6, 0.7, 0.8]))

        result = np.concatenate((left, right), axis=1)

        assert isinstance(result, ArraySample)
        assert_weights_equal(result, np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))

    def test_array_function_concatenate_fills_missing_weights_with_ones(self) -> None:
        left = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1)
        right = ArraySample(np.arange(12, 24).reshape((3, 4)), sample_axis=1, weights=np.array([0.5, 0.6, 0.7, 0.8]))

        result = np.concatenate((left, right), axis=1)

        assert isinstance(result, ArraySample)
        assert_weights_equal(result, np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.6, 0.7, 0.8]))

    def test_array_function_concatenate_weighted_non_sample_axis_raises(self) -> None:
        left = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=np.array([0.1, 0.2, 0.3, 0.4]))
        right = ArraySample(np.arange(12, 24).reshape((3, 4)), sample_axis=1, weights=np.array([0.5, 0.6, 0.7, 0.8]))

        with pytest.raises(ValueError, match="sample axis"):
            np.concatenate((left, right), axis=0)

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

    def test_array_function_stack_with_weights_raises(self) -> None:
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        left = ArraySample(np.arange(12).reshape((3, 4)), sample_axis=1, weights=weights)
        right = ArraySample(np.arange(12, 24).reshape((3, 4)), sample_axis=1, weights=weights)

        with pytest.raises(ValueError, match="stack"):
            np.stack((left, right), axis=0)

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
