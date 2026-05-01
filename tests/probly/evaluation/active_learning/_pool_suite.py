"""Shared pool test suite for all backends."""

from __future__ import annotations

import pytest

from probly.evaluation.active_learning.pool import from_dataset, query


class PoolSuite:
    """Backend-agnostic pool tests.

    Requires fixtures: classification_data, arrays_equal, concat_fn, sort_fn,
    copy_fn, index_fn.
    """

    def test_from_dataset_labeled_size(self, classification_data):
        x_train, y_train, x_test, y_test = classification_data
        pool = from_dataset(x_train, y_train, x_test, y_test, initial_size=20, seed=0)
        assert pool.n_labeled == 20

    def test_from_dataset_unlabeled_size(self, classification_data):
        x_train, y_train, x_test, y_test = classification_data
        pool = from_dataset(x_train, y_train, x_test, y_test, initial_size=20, seed=0)
        assert pool.n_unlabeled == 130

    def test_from_dataset_total_accounts_for_all(self, classification_data):
        x_train, y_train, x_test, y_test = classification_data
        pool = from_dataset(x_train, y_train, x_test, y_test, initial_size=30, seed=0)
        assert pool.n_labeled + pool.n_unlabeled == 150

    def test_from_dataset_no_overlap(self, classification_data, concat_fn, sort_fn, arrays_equal):
        x_train, y_train, x_test, y_test = classification_data
        pool = from_dataset(x_train, y_train, x_test, y_test, initial_size=40, seed=0)
        x_all = concat_fn([pool.x_labeled, pool.x_unlabeled])
        assert len(x_all) == len(x_train)
        assert arrays_equal(sort_fn(x_train), sort_fn(x_all))

    def test_from_dataset_same_seed_same_result(self, classification_data, arrays_equal):
        x_train, y_train, x_test, y_test = classification_data
        pool1 = from_dataset(x_train, y_train, x_test, y_test, initial_size=20, seed=7)
        pool2 = from_dataset(x_train, y_train, x_test, y_test, initial_size=20, seed=7)
        assert arrays_equal(pool1.x_labeled, pool2.x_labeled)
        assert arrays_equal(pool1.x_unlabeled, pool2.x_unlabeled)

    def test_from_dataset_different_seeds_different_results(self, classification_data, arrays_equal):
        x_train, y_train, x_test, y_test = classification_data
        pool1 = from_dataset(x_train, y_train, x_test, y_test, initial_size=20, seed=1)
        pool2 = from_dataset(x_train, y_train, x_test, y_test, initial_size=20, seed=2)
        assert not arrays_equal(pool1.x_labeled, pool2.x_labeled)

    def test_query_labeled_grows(self, classification_data, index_fn):
        x_train, y_train, x_test, y_test = classification_data
        pool = from_dataset(x_train, y_train, x_test, y_test, initial_size=20, seed=0)
        query(pool, index_fn([0, 1, 2, 3, 4]))
        assert pool.n_labeled == 25

    def test_query_unlabeled_shrinks(self, classification_data, index_fn):
        x_train, y_train, x_test, y_test = classification_data
        pool = from_dataset(x_train, y_train, x_test, y_test, initial_size=20, seed=0)
        query(pool, index_fn([0, 1, 2, 3, 4]))
        assert pool.n_unlabeled == 125

    def test_query_total_unchanged(self, classification_data, index_fn):
        x_train, y_train, x_test, y_test = classification_data
        pool = from_dataset(x_train, y_train, x_test, y_test, initial_size=20, seed=0)
        query(pool, index_fn([3, 7, 10]))
        assert pool.n_labeled + pool.n_unlabeled == 150

    def test_query_label_alignment(self, classification_data, copy_fn, arrays_equal, index_fn):
        x_train, y_train, x_test, y_test = classification_data
        pool = from_dataset(x_train, y_train, x_test, y_test, initial_size=20, seed=0)
        queried_x = copy_fn(pool.x_unlabeled[[0, 1, 2]])
        queried_y = copy_fn(pool.y_unlabeled[[0, 1, 2]])
        query(pool, index_fn([0, 1, 2]))
        assert arrays_equal(pool.x_labeled[-3:], queried_x)
        assert arrays_equal(pool.y_labeled[-3:], queried_y)

    def test_n_labeled_property(self, classification_data):
        x_train, y_train, x_test, y_test = classification_data
        pool = from_dataset(x_train, y_train, x_test, y_test, initial_size=50, seed=0)
        assert pool.n_labeled == len(pool.x_labeled)

    def test_n_unlabeled_property(self, classification_data):
        x_train, y_train, x_test, y_test = classification_data
        pool = from_dataset(x_train, y_train, x_test, y_test, initial_size=50, seed=0)
        assert pool.n_unlabeled == len(pool.x_unlabeled)

    def test_from_dataset_rejects_zero_initial_size(self, classification_data):
        """initial_size=0 must raise ValueError."""
        x_train, y_train, x_test, y_test = classification_data
        with pytest.raises(ValueError, match="initial_size must be in"):
            from_dataset(x_train, y_train, x_test, y_test, initial_size=0, seed=0)

    def test_from_dataset_rejects_oversized_initial(self, classification_data):
        """initial_size >= len(x) must raise ValueError."""
        x_train, y_train, x_test, y_test = classification_data
        with pytest.raises(ValueError, match="initial_size must be in"):
            from_dataset(x_train, y_train, x_test, y_test, initial_size=150, seed=0)

    def test_query_empty_indices(self, classification_data, index_fn):
        """Querying with empty indices should be a no-op."""
        x_train, y_train, x_test, y_test = classification_data
        pool = from_dataset(x_train, y_train, x_test, y_test, initial_size=20, seed=0)
        before = pool.n_labeled
        # Use index_fn([0])[:0] to get an empty integer-typed array (index_fn([])
        # may default to float dtype in some backends).
        empty = index_fn([0])[:0]
        query(pool, empty)
        assert pool.n_labeled == before
