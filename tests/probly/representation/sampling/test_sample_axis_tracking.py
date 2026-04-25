"""Tests for tracking the position of a special axis through numpy indexing operations."""

from __future__ import annotations

import numpy as np
import pytest

from probly.representation.array_like import ToIndices
from probly.representation.sample.axis_tracking import track_axis as track_axis_result


def track_axis(index: ToIndices, special_axis: int, ndim: int, torch_indexing: bool = False) -> object:
    result = track_axis_result(index, special_axis, ndim, torch_indexing=torch_indexing)
    return None if result is None else result.new_axis


def weight_index(index: ToIndices, special_axis: int, ndim: int, torch_indexing: bool = False) -> object:
    result = track_axis_result(index, special_axis, ndim, torch_indexing=torch_indexing)
    assert result is not None
    return result.index


class TestBasicIndexing:
    def test_basic_slice_preserves_axis(self) -> None:
        assert track_axis((slice(None), slice(None)), 0, 2) == 0
        assert track_axis((slice(None), slice(None)), 1, 5) == 1

    def test_basic_index_before_special(self) -> None:
        assert track_axis((2, slice(None)), 1, 2) == 0
        assert track_axis((2, 3, slice(None)), 2, 3) == 0

    def test_basic_index_after_special(self) -> None:
        assert track_axis((slice(None), 2), 0, 2) == 0
        assert track_axis((slice(None), slice(None), slice(None), 3, 4), 2, 5) == 2

    def test_special_axis_removed(self) -> None:
        assert track_axis((slice(None), 3), special_axis=1, ndim=2) is None
        assert track_axis((3, slice(None), 4), special_axis=2, ndim=4) is None


class TestNewAxis:
    def test_newaxis_before_special_shifts_up(self) -> None:
        assert track_axis((None, slice(None), slice(None)), 1, 2) == 2

    def test_newaxis_after_special_does_not_shift(self) -> None:
        assert track_axis((slice(None), None, slice(None)), 0, 2) == 0


class TestEllipsis:
    def test_ellipsis_expansion(self) -> None:
        assert track_axis((Ellipsis, slice(None)), special_axis=0, ndim=3) == 0
        assert track_axis((Ellipsis, slice(None)), special_axis=1, ndim=3) == 1

    def test_ellipsis_middle(self) -> None:
        assert track_axis((slice(None), Ellipsis, slice(None)), special_axis=0, ndim=4) == 0
        assert track_axis((slice(None), Ellipsis, slice(None)), special_axis=1, ndim=4) == 1
        assert track_axis((slice(None), Ellipsis, slice(None)), special_axis=3, ndim=4) == 3

    def test_ellipsis_with_axis_removed(self) -> None:
        assert track_axis((..., 10), special_axis=2, ndim=3) is None
        assert track_axis((slice(None), Ellipsis, 3), special_axis=3, ndim=4) is None

    def test_ellipsis_with_trailing_newaxis_preserves_last_axis(self) -> None:
        idx = (Ellipsis, None)
        assert track_axis(idx, 3, 4) == 3

    def test_ellipsis_with_trailing_1d_integer_index(self) -> None:
        idx = (Ellipsis, np.array([0, 2]))
        assert track_axis(idx, 0, 3) == 0
        assert track_axis(idx, 1, 3) == 1
        assert track_axis(idx, 2, 3) == 2

    def test_ellipsis_with_nd_integer_index_and_slice(self) -> None:
        idx = (np.array([[0, 1], [1, 0]]), Ellipsis, slice(None))
        assert track_axis(idx, 0, 3) is None
        assert track_axis(idx, 1, 3) == 2
        assert track_axis(idx, 2, 3) == 3

    def test_ellipsis_with_nd_boolean_index(self) -> None:
        idx = (np.array([[True, False, True], [False, True, False]]), Ellipsis)
        assert track_axis(idx, 0, 3) == 0
        assert track_axis(idx, 1, 3) == 0
        assert track_axis(idx, 2, 3) == 1

    def test_ellipsis_with_leading_newaxis_and_trailing_1d_integer_index(self) -> None:
        idx = (None, Ellipsis, np.array([0, 2]))
        assert track_axis(idx, 0, 3) == 1
        assert track_axis(idx, 1, 3) == 2
        assert track_axis(idx, 2, 3) == 3

    def test_ellipsis_with_1d_integer_index_newaxis_and_slice(self) -> None:
        idx = (np.array([0, 1]), Ellipsis, None, slice(None))
        assert track_axis(idx, 0, 3) == 0
        assert track_axis(idx, 1, 3) == 1
        assert track_axis(idx, 2, 3) == 3

    def test_ellipsis_with_slice_and_trailing_1d_boolean_index(self) -> None:
        idx = (slice(None), Ellipsis, np.array([True, False, True, False]))
        assert track_axis(idx, 0, 3) == 0
        assert track_axis(idx, 1, 3) == 1
        assert track_axis(idx, 2, 3) == 2

    def test_trailing_scalar_boolean_with_ellipsis_keeps_last_axis_position(self) -> None:
        idx = (Ellipsis, True)
        assert track_axis(idx, 0, 4) == 0
        assert track_axis(idx, 1, 4) == 1
        assert track_axis(idx, 2, 4) == 2
        assert track_axis(idx, 3, 4) == 3

    def test_trailing_numpy_scalar_boolean_with_ellipsis_keeps_last_axis_position(self) -> None:
        idx = (Ellipsis, np.array(True))
        assert track_axis(idx, 0, 4) == 0
        assert track_axis(idx, 1, 4) == 1
        assert track_axis(idx, 2, 4) == 2
        assert track_axis(idx, 3, 4) == 3


class TestAdvancedIndexing:
    def test_advanced_index_keeps_special_subset(self) -> None:
        idx = (slice(None), np.array([1, 2]), slice(None))
        assert track_axis(idx, special_axis=1, ndim=3) == 1

    def test_higher_dim_advanced_index_discards_special(self) -> None:
        idx = (slice(None), np.array([[1, 2]]), slice(None))
        assert track_axis(idx, special_axis=1, ndim=3) is None

    def test_advanced_index_on_non_special_axis(self) -> None:
        idx = (np.array([0, 1]), slice(None), slice(None))
        assert track_axis(idx, 1, 3) == 1

    def test_advanced_higher_dim_index_on_non_special_axis(self) -> None:
        idx = (np.array([[0, 1]]), slice(None), slice(None))
        assert track_axis(idx, 1, 3) == 2

    def test_multiple_contiguous_advanced_indices_moves_special_axis(self) -> None:
        idx = (np.array([0, 1]), np.array([1, 2]), slice(None))
        assert track_axis(idx, 2, 3) == 1

    def test_multiple_contiguous_higher_dim_advanced_indices_moves_special_axis(self) -> None:
        idx = (np.array([1]), np.array([[[1], [2]]]), slice(None))
        assert track_axis(idx, 2, 3) == 3

    def test_multiple_noncontiguous_advanced_indices(self) -> None:
        idx = (slice(None), np.array([0, 1]), slice(None), np.array([1, 2]), slice(None))
        assert track_axis(idx, 0, 5) == 1

    def test_multiple_noncontiguous_higher_dim_advanced_indices(self) -> None:
        idx = (slice(None), np.array([[0, 1]]), slice(None), np.array([[1, 2]]), slice(None))
        assert track_axis(idx, 0, 5) == 2

    def test_advanced_boolean_indexing_keeps_special_axis(self) -> None:
        idx = (slice(None), np.array([True, False, True]), slice(None))
        assert track_axis(idx, 1, 3) == 1

    def test_advanced_higher_dim_boolean_indexing_keeps_special_axis(self) -> None:
        idx = ([[True], [False]], slice(None))
        assert track_axis(idx, 1, 3) == 0

    def test_advanced_higher_dim_boolean_indexing_consumes_axes(self) -> None:
        idx = ([[True], [False]], 7)
        assert track_axis(idx, 2, 3) is None

    def test_scalar_boolean_index_adds_leading_axis(self) -> None:
        assert track_axis(True, 0, 3) == 1
        assert track_axis(True, 1, 3) == 2
        assert track_axis(True, 2, 3) == 3

    def test_scalar_numpy_boolean_index_adds_leading_axis(self) -> None:
        idx = np.array(True)
        assert track_axis(idx, 0, 3) == 1
        assert track_axis(idx, 1, 3) == 2
        assert track_axis(idx, 2, 3) == 3

    def test_scalar_boolean_index_inside_tuple_adds_axis_in_place(self) -> None:
        idx = (slice(None), True, slice(None))
        assert track_axis(idx, 0, 3) == 0
        assert track_axis(idx, 1, 3) == 2
        assert track_axis(idx, 2, 3) == 3

    def test_scalar_numpy_boolean_index_inside_tuple_adds_axis_in_place(self) -> None:
        idx = (slice(None), np.array(True), slice(None))
        assert track_axis(idx, 0, 3) == 0
        assert track_axis(idx, 1, 3) == 2
        assert track_axis(idx, 2, 3) == 3

    def test_noncontiguous_advanced_indices_place_indexed_axis_at_front(self) -> None:
        idx = (np.array([0, 2]), slice(None), np.array([0, 2]), slice(None))
        assert track_axis(idx, 0, 4) == 0
        assert track_axis(idx, 2, 4) == 0

    @pytest.mark.skip(reason="Mixed-rank advanced indexing behavior is intentionally deferred.")
    def test_mixed_rank_advanced_indexing_tracks_indexed_axis_position(self) -> None:
        idx = (np.array([[0, 1], [2, 0]]), slice(None), np.array([0, 2]), slice(None))
        assert track_axis(idx, 2, 4) == 1


class TestMixedIndexing:
    def test_insert_with_subset(self) -> None:
        idx = (
            None,  # new axis at front
            slice(None),
            np.array([1, 2]),  # advanced
            3,  # integer removes last original axis
        )
        assert track_axis(idx, 0, 3) == 1
        assert track_axis(idx, 1, 3) == 2
        assert track_axis(idx, 2, 3) is None

    def test_insert_with_noncontiguous_subset(self) -> None:
        idx = (
            None,  # new axis at front
            slice(None),
            np.array([[1, 2]]),  # advanced
            None,  # new axis after axis 0
            np.array([[3], [4]]),  # advanced
            slice(None),
        )
        assert track_axis(idx, 0, 4) == 3
        assert track_axis(idx, 1, 4) is None
        assert track_axis(idx, 2, 4) is None
        assert track_axis(idx, 3, 4) == 5


class TestIndexingModeSemantics:
    def test_torch_indexing_flag_changes_mixed_int_and_advanced_behavior(self) -> None:
        idx = (0, slice(None), np.array([0, 2]))

        # NumPy/JAX behavior
        assert track_axis(idx, special_axis=1, ndim=3, torch_indexing=False) == 1
        assert track_axis(idx, special_axis=2, ndim=3, torch_indexing=False) == 0

        # Torch behavior
        assert track_axis(idx, special_axis=1, ndim=3, torch_indexing=True) == 0
        assert track_axis(idx, special_axis=2, ndim=3, torch_indexing=True) == 1

    def test_empty_ellipsis_changes_advanced_index_separation(self) -> None:
        idx = (slice(None), np.array([0, 1]), Ellipsis, np.array([0, 1]))

        # NumPy/JAX behavior (NumPy PR parity case): empty ellipsis still separates advanced indices.
        assert track_axis(idx, special_axis=0, ndim=3, torch_indexing=False) == 1
        assert track_axis(idx, special_axis=1, ndim=3, torch_indexing=False) == 0
        assert track_axis(idx, special_axis=2, ndim=3, torch_indexing=False) == 0

        # Torch behavior: empty ellipsis does not separate advanced indices.
        assert track_axis(idx, special_axis=0, ndim=3, torch_indexing=True) == 0
        assert track_axis(idx, special_axis=1, ndim=3, torch_indexing=True) == 1
        assert track_axis(idx, special_axis=2, ndim=3, torch_indexing=True) == 1


class TestBooleanScalarModeSemantics:
    def test_python_bool_scalar_is_not_treated_as_integer(self) -> None:
        assert track_axis(True, special_axis=0, ndim=2, torch_indexing=False) == 1
        assert track_axis(True, special_axis=1, ndim=2, torch_indexing=False) == 2

    def test_numpy_0d_bool_scalar_is_not_treated_as_integer(self) -> None:
        idx = np.array(True)
        assert track_axis(idx, special_axis=0, ndim=2, torch_indexing=False) == 1
        assert track_axis(idx, special_axis=1, ndim=2, torch_indexing=False) == 2


class TestWeightIndexTracking:
    def test_basic_slice_on_non_sample_axis_preserves_all_weights(self) -> None:
        assert weight_index((slice(1, 3), slice(None)), special_axis=1, ndim=2) == slice(None)

    def test_basic_slice_on_sample_axis_indexes_weights(self) -> None:
        assert weight_index((slice(None), slice(1, 3)), special_axis=1, ndim=2) == slice(1, 3)

    def test_ellipsis_expansion_indexes_sample_axis(self) -> None:
        assert weight_index((Ellipsis, slice(1, 3)), special_axis=2, ndim=3) == slice(1, 3)

    def test_integer_before_sample_axis_preserves_all_weights(self) -> None:
        assert weight_index((0, slice(None)), special_axis=1, ndim=2) == slice(None)

    def test_newaxis_before_sample_axis_preserves_all_weights(self) -> None:
        assert weight_index((None, slice(None), slice(None)), special_axis=1, ndim=2) == slice(None)

    def test_1d_integer_index_on_sample_axis_indexes_weights(self) -> None:
        idx = np.array([3, 1])

        result = weight_index((slice(None), idx), special_axis=1, ndim=2)

        assert np.array_equal(np.asarray(result), idx)

    def test_1d_boolean_index_on_sample_axis_indexes_weights(self) -> None:
        idx = np.array([True, False, True, False])

        result = weight_index((slice(None), idx), special_axis=1, ndim=2)

        assert np.array_equal(np.asarray(result), idx)

    def test_advanced_index_on_non_sample_axis_preserves_all_weights(self) -> None:
        assert weight_index((np.array([0, 2]), slice(None)), special_axis=1, ndim=2) == slice(None)

    def test_multidimensional_boolean_index_touching_sample_axis_is_unsupported(self) -> None:
        idx = np.array([[True, False], [False, True]])

        assert weight_index((idx, slice(None)), special_axis=0, ndim=3) is NotImplemented
