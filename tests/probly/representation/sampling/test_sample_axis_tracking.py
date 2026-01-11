"""Tests for tracking the position of a special axis through numpy indexing operations."""

from __future__ import annotations

import numpy as np

from probly.representation.sampling.array_sample_axis_tracking import track_axis


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
