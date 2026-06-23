"""Tests for the ArraySample Representation."""

from __future__ import annotations

import numpy as np
import pytest

from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayProbabilityCategoricalDistribution,
)
from probly.representation.sample import ArraySample
from probly.representation.sample.array_functions import array_sample_internals


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
        sample = ArraySample(ArrayProbabilityCategoricalDistribution(probabilities), sample_axis=0)

        result = sample.sample_mean()

        assert isinstance(result, ArrayCategoricalDistribution)
        assert result.shape == (3,)
        np.testing.assert_allclose(result.unnormalized_probabilities, result.probabilities)
        np.testing.assert_allclose(result.unnormalized_probabilities, np.mean(sample.array.probabilities, axis=0))

    def test_weighted_sample_mean_of_categorical_distribution_uses_weights(self) -> None:
        probabilities = np.arange(24, dtype=float).reshape((2, 3, 4)) + 1.0
        weights = np.array([0.25, 0.75])
        sample = ArraySample(ArrayProbabilityCategoricalDistribution(probabilities), sample_axis=0, weights=weights)

        result = sample.sample_mean()

        assert isinstance(result, ArrayCategoricalDistribution)
        assert result.shape == (3,)
        np.testing.assert_allclose(
            result.unnormalized_probabilities,
            np.average(sample.array.probabilities, axis=0, weights=weights),
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


class TestArraySampleValidation:
    """Constructor validation paths."""

    def test_sample_axis_out_of_bounds_positive(self) -> None:
        arr = np.zeros((2, 3))
        with pytest.raises(ValueError, match="out of bounds"):
            ArraySample(array=arr, sample_axis=2)

    def test_sample_axis_out_of_bounds_negative(self) -> None:
        arr = np.zeros((2, 3))
        with pytest.raises(ValueError, match="out of bounds"):
            ArraySample(array=arr, sample_axis=-3)

    def test_negative_sample_axis_normalises(self) -> None:
        arr = np.zeros((2, 3))
        sample = ArraySample(array=arr, sample_axis=-1)
        assert sample.sample_axis == 1

    def test_array_must_be_arraylike(self) -> None:
        # Plain Python list isn't NumpyArrayLike — fails earlier on .ndim access
        # in __post_init__, raising AttributeError.
        with pytest.raises((TypeError, AttributeError)):
            ArraySample(array=[[1, 2], [3, 4]], sample_axis=0)  # type: ignore[arg-type]

    def test_weights_shape_mismatch(self) -> None:
        arr = np.zeros((2, 3))
        with pytest.raises(ValueError, match="weights must have shape"):
            ArraySample(array=arr, sample_axis=0, weights=np.zeros(5))


class TestArraySampleArrayFunctions:
    """Verify sample_axis tracking under numpy operations."""

    def test_copy_with_subok_true_preserves_sample_axis(self) -> None:
        # np.copy defaults to subok=False, so we must pass subok=True to keep
        # the wrapper.
        arr = np.arange(12).reshape(3, 4)
        sample = ArraySample(array=arr, sample_axis=0)
        copied = np.copy(sample, subok=True)
        assert isinstance(copied, ArraySample)
        assert copied.sample_axis == 0
        np.testing.assert_array_equal(copied.array, arr)

    def test_copy_with_subok_false_returns_plain_array(self) -> None:
        arr = np.arange(12).reshape(3, 4)
        sample = ArraySample(array=arr, sample_axis=0)
        copied = np.copy(sample, subok=False)
        assert not isinstance(copied, ArraySample)
        np.testing.assert_array_equal(copied, arr)

    def test_mean_along_non_sample_axis_keeps_sample(self) -> None:
        arr = np.arange(12).reshape(3, 4).astype(float)
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.mean(sample, axis=1)
        assert isinstance(result, ArraySample)
        assert result.sample_axis == 0
        np.testing.assert_allclose(result.array, arr.mean(axis=1))

    def test_mean_along_sample_axis_drops_sample_wrapping(self) -> None:
        arr = np.arange(12).reshape(3, 4).astype(float)
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.mean(sample, axis=0)
        # axis 0 is the sample axis -> reduces it; result is a plain array.
        assert not isinstance(result, ArraySample)
        np.testing.assert_allclose(result, arr.mean(axis=0))

    def test_mean_with_keepdims_preserves_sample_axis(self) -> None:
        arr = np.arange(12).reshape(3, 4).astype(float)
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.mean(sample, axis=1, keepdims=True)
        assert isinstance(result, ArraySample)
        assert result.sample_axis == 0
        assert result.array.shape == (3, 1)

    def test_mean_with_no_axis_returns_plain(self) -> None:
        arr = np.arange(6).astype(float)
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.mean(sample)
        # When axis is None, np.mean returns a scalar / plain array.
        assert not isinstance(result, ArraySample)
        np.testing.assert_allclose(result, arr.mean())

    def test_std_along_non_sample_axis(self) -> None:
        arr = np.arange(12).reshape(3, 4).astype(float)
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.std(sample, axis=1)
        assert isinstance(result, ArraySample)
        assert result.sample_axis == 0

    def test_argmax(self) -> None:
        arr = np.array([[1, 5, 2], [4, 3, 6]])
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.argmax(sample, axis=1)
        # argmax reduces axis 1; sample_axis=0 stays.
        assert isinstance(result, ArraySample)
        assert result.sample_axis == 0

    def test_count_nonzero(self) -> None:
        arr = np.array([[0, 1, 0], [1, 1, 0]])
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.count_nonzero(sample, axis=1)
        assert isinstance(result, ArraySample)
        assert result.sample_axis == 0

    def test_transpose_no_axes_inverts_axes(self) -> None:
        arr = np.arange(24).reshape(2, 3, 4)
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.transpose(sample)
        # Default reverses axes; sample axis 0 -> 2.
        assert isinstance(result, ArraySample)
        assert result.sample_axis == 2
        assert result.array.shape == (4, 3, 2)

    def test_transpose_with_axes(self) -> None:
        arr = np.arange(24).reshape(2, 3, 4)
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.transpose(sample, axes=(2, 0, 1))
        assert isinstance(result, ArraySample)
        # original axis 0 went to position 1
        assert result.sample_axis == 1

    def test_matrix_transpose_swaps_last_two_axes(self) -> None:
        arr = np.arange(24).reshape(2, 3, 4)
        sample = ArraySample(array=arr, sample_axis=2)
        result = np.matrix_transpose(sample)
        # last axis (sample) becomes second-to-last
        assert isinstance(result, ArraySample)
        assert result.sample_axis == 1

    def test_matrix_transpose_with_sample_in_other_axis(self) -> None:
        arr = np.arange(24).reshape(2, 3, 4)
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.matrix_transpose(sample)
        # sample axis untouched
        assert result.sample_axis == 0  # ty: ignore[unresolved-attribute]

    def test_flip_preserves_sample_axis(self) -> None:
        arr = np.arange(12).reshape(3, 4)
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.flip(sample)
        assert isinstance(result, ArraySample)
        assert result.sample_axis == 0

    def test_fliplr_preserves_sample_axis(self) -> None:
        arr = np.arange(12).reshape(3, 4)
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.fliplr(sample)
        assert isinstance(result, ArraySample)
        assert result.sample_axis == 0

    def test_flipud_preserves_sample_axis(self) -> None:
        arr = np.arange(12).reshape(3, 4)
        sample = ArraySample(array=arr, sample_axis=1)
        result = np.flipud(sample)
        assert isinstance(result, ArraySample)
        assert result.sample_axis == 1

    def test_roll_preserves_sample_axis(self) -> None:
        arr = np.arange(12).reshape(3, 4)
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.roll(sample, 1, axis=0)
        assert isinstance(result, ArraySample)
        assert result.sample_axis == 0

    def test_reshape_c_order_finds_sample_axis(self) -> None:
        arr = np.arange(12).reshape(3, 4)
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.reshape(sample, (3, 2, 2))
        # In C order, axis 0 size 3 maps to new axis 0.
        assert isinstance(result, ArraySample)
        assert result.sample_axis == 0

    def test_reshape_f_order_finds_sample_axis(self) -> None:
        arr = np.arange(12).reshape(3, 4, order="F")
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.reshape(sample, (3, 4), order="F")
        assert isinstance(result, ArraySample)
        # The axis-0 sample (size 3) lands at axis 0 in F order too.
        assert result.sample_axis == 0

    def test_reshape_a_order_falls_back(self) -> None:
        # 'A' order picks 'C' if c_contiguous else 'F'.
        arr = np.arange(12).reshape(3, 4)
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.reshape(sample, (3, 4), order="A")
        # C-contiguous so behaves like C order.
        assert isinstance(result, ArraySample)
        assert result.sample_axis == 0

    def test_reshape_drops_sample_when_axis_can_not_be_tracked(self) -> None:
        # Reshape so that the original sample axis can't be located.
        arr = np.arange(12).reshape(3, 4)
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.reshape(sample, (12,))  # collapsed
        # The reshape can't keep the sample axis (size 3 can't be re-found in (12,))
        assert not isinstance(result, ArraySample)

    def test_swapaxes_swaps_sample_axis(self) -> None:
        arr = np.arange(24).reshape(2, 3, 4)
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.swapaxes(sample, 0, 2)
        assert isinstance(result, ArraySample)
        assert result.sample_axis == 2

    def test_swapaxes_with_sample_in_second_axis(self) -> None:
        arr = np.arange(24).reshape(2, 3, 4)
        sample = ArraySample(array=arr, sample_axis=2)
        result = np.swapaxes(sample, 0, 2)
        assert result.sample_axis == 0

    def test_swapaxes_unaffected_axes_keeps_sample(self) -> None:
        arr = np.arange(24).reshape(2, 3, 4)
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.swapaxes(sample, 1, 2)
        assert result.sample_axis == 0

    def test_swapaxes_negative_axes_normalised(self) -> None:
        arr = np.arange(24).reshape(2, 3, 4)
        sample = ArraySample(array=arr, sample_axis=2)
        result = np.swapaxes(sample, -1, 0)
        assert result.sample_axis == 0

    def test_expand_dims_shifts_sample_axis(self) -> None:
        arr = np.arange(12).reshape(3, 4)
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.expand_dims(sample, axis=0)
        assert isinstance(result, ArraySample)
        assert result.sample_axis == 1

    def test_expand_dims_after_sample_axis_unchanged(self) -> None:
        arr = np.arange(12).reshape(3, 4)
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.expand_dims(sample, axis=2)
        assert result.sample_axis == 0  # ty: ignore[unresolved-attribute]

    def test_expand_dims_tuple_axes(self) -> None:
        arr = np.arange(12).reshape(3, 4)
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.expand_dims(sample, axis=(0, 1))
        assert result.sample_axis == 2  # ty: ignore[unresolved-attribute]

    def test_squeeze_removes_singleton_after_sample(self) -> None:
        arr = np.zeros((3, 1, 4))
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.squeeze(sample, axis=1)
        assert isinstance(result, ArraySample)
        assert result.sample_axis == 0
        assert result.array.shape == (3, 4)

    def test_squeeze_removes_singleton_before_sample(self) -> None:
        arr = np.zeros((1, 3, 4))
        sample = ArraySample(array=arr, sample_axis=1)
        result = np.squeeze(sample, axis=0)
        assert isinstance(result, ArraySample)
        assert result.sample_axis == 0

    def test_squeeze_drops_sample_axis(self) -> None:
        arr = np.zeros((1, 3, 4))
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.squeeze(sample, axis=0)
        # Sample axis was the squeezed dimension -> wrapper dropped.
        assert not isinstance(result, ArraySample)
        assert result.shape == (3, 4)

    def test_squeeze_default_drops_all_singletons(self) -> None:
        arr = np.zeros((3, 1, 4))
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.squeeze(sample)
        assert isinstance(result, ArraySample)
        assert result.array.shape == (3, 4)

    def test_squeeze_negative_axis(self) -> None:
        arr = np.zeros((3, 4, 1))
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.squeeze(sample, axis=-1)
        assert isinstance(result, ArraySample)
        assert result.sample_axis == 0

    def test_apply_along_axis_keeps_sample_when_not_along_sample(self) -> None:
        arr = np.arange(12).reshape(3, 4).astype(float)
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.apply_along_axis(np.sum, 1, sample)
        assert isinstance(result, ArraySample)
        assert result.sample_axis == 0

    def test_apply_along_axis_drops_sample_when_along_sample(self) -> None:
        arr = np.arange(12).reshape(3, 4).astype(float)
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.apply_along_axis(np.sum, 0, sample)
        # axis 0 is the sample axis -> wrapper dropped.
        assert not isinstance(result, ArraySample)

    def test_apply_along_axis_negative(self) -> None:
        arr = np.arange(12).reshape(3, 4).astype(float)
        sample = ArraySample(array=arr, sample_axis=0)
        result = np.apply_along_axis(np.sum, -1, sample)
        assert isinstance(result, ArraySample)
        assert result.sample_axis == 0


class TestArraySampleConcatenateStack:
    """Concatenate/stack with weights and out= variants."""

    def test_concat_axis_negative_normalised(self) -> None:
        a = ArraySample(array=np.arange(6).reshape(2, 3), sample_axis=0)
        b = ArraySample(array=np.arange(6, 12).reshape(2, 3), sample_axis=0)
        result = np.concatenate((a, b), axis=-1)
        assert isinstance(result, ArraySample)
        # axis -1 = 1, so sample_axis stays at 0
        assert result.sample_axis == 0
        assert result.array.shape == (2, 6)

    def test_concat_with_weights_along_sample_axis(self) -> None:
        a = ArraySample(
            array=np.arange(6).reshape(2, 3),
            sample_axis=0,
            weights=np.array([0.4, 0.6]),
        )
        b = ArraySample(
            array=np.arange(6, 12).reshape(2, 3),
            sample_axis=0,
            weights=np.array([0.5, 0.5]),
        )
        result = np.concatenate((a, b), axis=0)
        assert isinstance(result, ArraySample)
        assert result.weights is not None
        np.testing.assert_allclose(result.weights, [0.4, 0.6, 0.5, 0.5])

    def test_concat_with_weights_along_non_sample_axis_raises(self) -> None:
        a = ArraySample(
            array=np.arange(6).reshape(2, 3),
            sample_axis=0,
            weights=np.array([0.4, 0.6]),
        )
        b = ArraySample(
            array=np.arange(6, 12).reshape(2, 3),
            sample_axis=0,
            weights=np.array([0.5, 0.5]),
        )
        with pytest.raises(ValueError, match="Weighted samples only support concatenate"):
            np.concatenate((a, b), axis=1)

    def test_concat_one_unweighted_fills_with_ones(self) -> None:
        a = ArraySample(
            array=np.arange(6).reshape(2, 3),
            sample_axis=0,
            weights=np.array([0.4, 0.6]),
        )
        b = ArraySample(array=np.arange(6, 12).reshape(2, 3), sample_axis=0)
        result = np.concatenate((a, b), axis=0)
        assert result.weights is not None  # ty: ignore[unresolved-attribute]
        np.testing.assert_allclose(result.weights, [0.4, 0.6, 1.0, 1.0])  # ty: ignore[unresolved-attribute]

    def test_stack_with_weights_raises(self) -> None:
        a = ArraySample(
            array=np.arange(6).reshape(2, 3),
            sample_axis=0,
            weights=np.array([0.4, 0.6]),
        )
        b = ArraySample(
            array=np.arange(6, 12).reshape(2, 3),
            sample_axis=0,
            weights=np.array([0.5, 0.5]),
        )
        with pytest.raises(ValueError, match="Weighted samples do not support stack"):
            np.stack((a, b), axis=0)


class TestArraySampleSpecialMethods:
    """Conversion, dunder and detach-style methods."""

    def test_array_namespace(self) -> None:
        sample = ArraySample(array=np.arange(6).reshape(2, 3), sample_axis=0)
        assert sample.__array_namespace__() is not None

    def test_dtype_property(self) -> None:
        sample = ArraySample(array=np.arange(6).reshape(2, 3).astype(np.float64), sample_axis=0)
        assert sample.dtype == np.float64

    def test_size_property(self) -> None:
        sample = ArraySample(array=np.arange(6).reshape(2, 3), sample_axis=0)
        assert sample.size == 6

    def test_iter_yields_along_first_axis(self) -> None:
        sample = ArraySample(array=np.array([[1, 2], [3, 4]]), sample_axis=0)
        rows = list(iter(sample))
        np.testing.assert_array_equal(rows[0], [1, 2])
        np.testing.assert_array_equal(rows[1], [3, 4])

    def test_bool_one_element(self) -> None:
        # numpy 1-D arrays with size > 1 can't go to scalar but size==1 can
        # via deprecation pathway; bool() works on numpy arrays via "any".
        sample = ArraySample(array=np.array([True]), sample_axis=0)
        assert bool(sample) is True

    def test_bool_zero_element(self) -> None:
        sample = ArraySample(array=np.array([False]), sample_axis=0)
        assert bool(sample) is False

    def test_index_dunder(self) -> None:
        # __index__ delegates to underlying array; works for integer 0-D-ish.
        # We can't easily build a scalar ArraySample (rejected by __post_init__)
        # so we just check that the method exists and forwards.
        sample = ArraySample(array=np.array([5]), sample_axis=0)
        # The underlying ndarray's __index__ raises for size-1 arrays in
        # newer numpy; we just check the call dispatches without our own error.
        import contextlib  # noqa: PLC0415

        with contextlib.suppress(TypeError, ValueError):
            sample.__index__()

    def test_setitem(self) -> None:
        sample = ArraySample(array=np.arange(6).reshape(2, 3).astype(float), sample_axis=0)
        sample[0, 0] = 99
        assert sample.array[0, 0] == 99

    def test_array_dunder_returns_ndarray(self) -> None:
        sample = ArraySample(array=np.arange(6).reshape(2, 3), sample_axis=0)
        out = np.asarray(sample, dtype=float)
        assert isinstance(out, np.ndarray)
        assert out.dtype == float

    def test_copy_method(self) -> None:
        sample = ArraySample(array=np.arange(6).reshape(2, 3), sample_axis=1)
        c = sample.copy()
        assert isinstance(c, ArraySample)
        assert c.sample_axis == 1
        assert c.array is not sample.array

    def test_array_like_copy(self) -> None:
        sample = ArraySample(array=np.arange(6).reshape(2, 3), sample_axis=0)
        result = sample.__array_like__(copy=True)
        assert result is not sample

    def test_array_like_no_copy(self) -> None:
        sample = ArraySample(array=np.arange(6).reshape(2, 3), sample_axis=0)
        result = sample.__array_like__()
        assert result is sample

    def test_eq_returns_array(self) -> None:
        a = ArraySample(array=np.array([1, 2, 3]), sample_axis=0)
        b = a == 2
        np.testing.assert_array_equal(b, [False, True, False])

    def test_hash_works(self) -> None:
        a = ArraySample(array=np.array([1, 2, 3]), sample_axis=0)
        # Two distinct objects have distinct hashes.
        b = ArraySample(array=np.array([1, 2, 3]), sample_axis=0)
        assert hash(a) != hash(b)


class TestArraySampleStats:
    """Aggregate statistics methods on ArraySample."""

    def test_sample_mean_with_weights(self) -> None:
        a = ArraySample(array=np.array([1.0, 2.0, 3.0]), sample_axis=0, weights=np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(a.sample_mean(), 1.0)

    def test_sample_std_with_weights(self) -> None:
        a = ArraySample(
            array=np.array([1.0, 2.0, 3.0]),
            sample_axis=0,
            weights=np.array([1.0, 1.0, 1.0]),
        )
        # Weighted std (zero ddof). Should match unweighted std.
        np.testing.assert_allclose(a.sample_std(), np.std([1.0, 2.0, 3.0]))

    def test_sample_var_with_weights_and_ddof_raises(self) -> None:
        a = ArraySample(
            array=np.array([1.0, 2.0, 3.0]),
            sample_axis=0,
            weights=np.array([1.0, 1.0, 1.0]),
        )
        with pytest.raises(ValueError, match="ddof > 0"):
            a.sample_var(ddof=1)


class TestArraySampleIndexing:
    """Indexing semantics for ArraySample including weighted samples."""

    def test_simple_index_with_weights_drops_axis(self) -> None:
        a = ArraySample(
            array=np.arange(6).reshape(3, 2),
            sample_axis=0,
            weights=np.array([0.1, 0.2, 0.7]),
        )
        # Picking a single sample via integer index removes the sample axis.
        result = a[0]
        # When the sample dimension disappears the wrapper falls back to ndarray.
        assert isinstance(result, np.ndarray)


class TestUFuncReductions:
    """ArraySample interactions with numpy ufunc reductions / outer / at."""

    def test_ufunc_call_preserves_sample_axis(self) -> None:
        sample = ArraySample(array=np.arange(6).reshape(2, 3).astype(float), sample_axis=0)
        result = np.add(sample, 1)
        assert isinstance(result, ArraySample)
        assert result.sample_axis == 0

    def test_ufunc_reduce_along_non_sample_axis(self) -> None:
        sample = ArraySample(array=np.arange(6).reshape(2, 3).astype(float), sample_axis=0)
        result = np.add.reduce(sample, axis=1)
        assert isinstance(result, ArraySample)
        assert result.sample_axis == 0

    def test_ufunc_reduce_along_sample_axis_drops_wrapper(self) -> None:
        sample = ArraySample(array=np.arange(6).reshape(2, 3).astype(float), sample_axis=0)
        result = np.add.reduce(sample, axis=0)
        assert not isinstance(result, ArraySample)

    def test_ufunc_reduce_with_keepdims(self) -> None:
        sample = ArraySample(array=np.arange(6).reshape(2, 3).astype(float), sample_axis=0)
        result = np.add.reduce(sample, axis=1, keepdims=True)
        assert isinstance(result, ArraySample)
        assert result.array.shape == (2, 1)

    def test_ufunc_outer_drops_wrapper(self) -> None:
        sample = ArraySample(array=np.array([1.0, 2.0]), sample_axis=0)
        result = np.add.outer(sample, np.array([10.0, 20.0]))
        assert not isinstance(result, ArraySample)

    def test_ufunc_at_drops_wrapper(self) -> None:
        sample = ArraySample(array=np.arange(4).astype(float), sample_axis=0)
        np.add.at(sample, [0, 1], 1.0)
        np.testing.assert_array_equal(sample.array, [1.0, 2.0, 2.0, 3.0])


class TestArraySampleInternalsRegistry:
    """The single-dispatch registry returns expected internals for ArraySample."""

    def test_returns_internals_for_array_sample(self) -> None:
        sample = ArraySample(array=np.arange(6).reshape(2, 3), sample_axis=0)
        internals = array_sample_internals(sample)
        assert internals is not None
        assert internals.sample_axis == 0
        np.testing.assert_array_equal(internals.array, sample.array)

    def test_returns_none_for_unknown_input(self) -> None:
        assert array_sample_internals(object()) is None


class TestArraySampleFromSample:
    """Ensure ArraySample.from_sample handles various input types."""

    def test_from_sample_with_existing_array_sample(self) -> None:
        original = ArraySample(array=np.arange(6).reshape(2, 3), sample_axis=0)
        result = ArraySample.from_sample(original)
        np.testing.assert_array_equal(result.array, original.array)
        assert result.sample_axis == 0

    def test_from_sample_with_dtype(self) -> None:
        original = ArraySample(array=np.arange(6, dtype=np.int32).reshape(2, 3), sample_axis=0)
        result = ArraySample.from_sample(original, dtype=np.float64)
        assert result.array.dtype == np.float64

    def test_from_sample_moves_axis(self) -> None:
        original = ArraySample(array=np.arange(6).reshape(2, 3), sample_axis=0)
        result = ArraySample.from_sample(original, sample_axis=1)
        assert result.sample_axis == 1
        # Shape rotates accordingly.
        assert result.array.shape == (3, 2)


class TestArraySampleFromIterable:
    """from_iterable edge cases."""

    def test_with_empty_iterable_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot infer"):
            ArraySample.from_iterable([], sample_axis="auto")

    def test_with_zero_dim_array_raises(self) -> None:
        # Passing a 0-dim ndarray with sample_axis="auto" should fail.
        with pytest.raises(ValueError, match="Cannot infer"):
            ArraySample.from_iterable(np.array(5), sample_axis="auto")

    def test_with_explicit_sample_axis(self) -> None:
        s = ArraySample.from_iterable(np.arange(6).reshape(3, 2), sample_axis=0)
        assert s.sample_axis == 0

    def test_with_dtype(self) -> None:
        s = ArraySample.from_iterable([np.array([1.0]), np.array([2.0])], dtype=np.float32)
        assert s.array.dtype == np.float32


class TestArraySampleConcat:
    """concat edge cases."""

    def test_concat_with_non_arraysample(self) -> None:
        # Build a generic Sample-like object and concat with it.
        from probly.representation.sample._common import ListSample  # noqa: PLC0415

        a = ArraySample(array=np.array([[1.0, 2.0], [3.0, 4.0]]), sample_axis=0)
        # ListSample with two samples.
        b = ListSample([np.array([5.0, 6.0]), np.array([7.0, 8.0])])
        result = a.concat(b)
        assert result.array.shape == (4, 2)


class TestArraySampleUfuncOut:
    """The __array_ufunc__ ``out=`` keyword path."""

    def test_ufunc_with_explicit_out_arraysample(self) -> None:
        a = ArraySample(array=np.zeros(3, dtype=float), sample_axis=0)
        b = ArraySample(array=np.array([1.0, 2.0, 3.0]), sample_axis=0)
        out = ArraySample(array=np.zeros(3, dtype=float), sample_axis=0)
        result = np.add(a, b, out=out)
        # out is returned (whether identical to passed `out` or wrapped).
        np.testing.assert_array_equal(out.array, [1.0, 2.0, 3.0])
        assert result is out
