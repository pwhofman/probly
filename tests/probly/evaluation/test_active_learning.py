"""Tests for the ActiveLearningPool data structure."""

from __future__ import annotations

import numpy as np
import pytest

from probly.evaluation.active_learning.pool import ActiveLearningPool

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def classification_data():
    """200 samples, 5 features, 3 classes, 150/50 train/test split."""
    rng = np.random.default_rng(42)
    n_total = 200
    n_features = 5
    n_classes = 3
    x = rng.standard_normal((n_total, n_features))
    y = rng.integers(0, n_classes, size=n_total)
    x_train, y_train = x[:150], y[:150]
    x_test, y_test = x[150:], y[150:]
    return x_train, y_train, x_test, y_test


# ---------------------------------------------------------------------------
# from_dataset: split sizes
# ---------------------------------------------------------------------------


def test_from_dataset_labeled_size(classification_data):
    x_train, y_train, x_test, y_test = classification_data
    pool = ActiveLearningPool.from_dataset(x_train, y_train, x_test, y_test, initial_size=20, seed=0)
    assert pool.n_labeled == 20


def test_from_dataset_unlabeled_size(classification_data):
    x_train, y_train, x_test, y_test = classification_data
    pool = ActiveLearningPool.from_dataset(x_train, y_train, x_test, y_test, initial_size=20, seed=0)
    assert pool.n_unlabeled == 130


def test_from_dataset_total_accounts_for_all(classification_data):
    x_train, y_train, x_test, y_test = classification_data
    pool = ActiveLearningPool.from_dataset(x_train, y_train, x_test, y_test, initial_size=30, seed=0)
    assert pool.n_labeled + pool.n_unlabeled == 150


# ---------------------------------------------------------------------------
# from_dataset: no overlap
# ---------------------------------------------------------------------------


def test_from_dataset_no_overlap(classification_data):
    """Every training sample appears in exactly one of labeled or unlabeled."""
    x_train, y_train, x_test, y_test = classification_data
    pool = ActiveLearningPool.from_dataset(x_train, y_train, x_test, y_test, initial_size=40, seed=0)
    # Reconstruct the full training set from the pool.
    x_all = np.concatenate([pool.x_labeled, pool.x_unlabeled], axis=0)
    # Each row of x_train must appear exactly once in x_all.
    assert x_all.shape[0] == x_train.shape[0]
    # Check that the sorted rows match (up to permutation).
    sorted_train = np.sort(x_train, axis=0)
    sorted_all = np.sort(x_all, axis=0)
    np.testing.assert_array_equal(sorted_train, sorted_all)


# ---------------------------------------------------------------------------
# from_dataset -- determinism
# ---------------------------------------------------------------------------


def test_from_dataset_same_seed_same_result(classification_data):
    x_train, y_train, x_test, y_test = classification_data
    pool1 = ActiveLearningPool.from_dataset(x_train, y_train, x_test, y_test, initial_size=20, seed=7)
    pool2 = ActiveLearningPool.from_dataset(x_train, y_train, x_test, y_test, initial_size=20, seed=7)
    np.testing.assert_array_equal(pool1.x_labeled, pool2.x_labeled)
    np.testing.assert_array_equal(pool1.x_unlabeled, pool2.x_unlabeled)


def test_from_dataset_different_seeds_different_results(classification_data):
    x_train, y_train, x_test, y_test = classification_data
    pool1 = ActiveLearningPool.from_dataset(x_train, y_train, x_test, y_test, initial_size=20, seed=1)
    pool2 = ActiveLearningPool.from_dataset(x_train, y_train, x_test, y_test, initial_size=20, seed=2)
    # With 150 samples and 20 labeled, the chance of identical splits is negligible.
    assert not np.array_equal(pool1.x_labeled, pool2.x_labeled)


# ---------------------------------------------------------------------------
# query: moves samples correctly
# ---------------------------------------------------------------------------


def test_query_labeled_grows(classification_data):
    x_train, y_train, x_test, y_test = classification_data
    pool = ActiveLearningPool.from_dataset(x_train, y_train, x_test, y_test, initial_size=20, seed=0)
    query_indices = np.array([0, 1, 2, 3, 4])
    pool.query(query_indices)
    assert pool.n_labeled == 25


def test_query_unlabeled_shrinks(classification_data):
    x_train, y_train, x_test, y_test = classification_data
    pool = ActiveLearningPool.from_dataset(x_train, y_train, x_test, y_test, initial_size=20, seed=0)
    query_indices = np.array([0, 1, 2, 3, 4])
    pool.query(query_indices)
    assert pool.n_unlabeled == 125


def test_query_total_unchanged(classification_data):
    """Total number of training samples is conserved after a query."""
    x_train, y_train, x_test, y_test = classification_data
    pool = ActiveLearningPool.from_dataset(x_train, y_train, x_test, y_test, initial_size=20, seed=0)
    query_indices = np.array([3, 7, 10])
    pool.query(query_indices)
    assert pool.n_labeled + pool.n_unlabeled == 150


# ---------------------------------------------------------------------------
# query: label alignment preserved
# ---------------------------------------------------------------------------


def test_query_label_alignment(classification_data):
    """After a query, y_labeled[-k:] matches the y values of the queried x rows."""
    x_train, y_train, x_test, y_test = classification_data
    pool = ActiveLearningPool.from_dataset(x_train, y_train, x_test, y_test, initial_size=20, seed=0)
    # Record which x rows are about to be queried.
    queried_x = pool.x_unlabeled[[0, 1, 2]].copy()
    queried_y = pool.y_unlabeled[[0, 1, 2]].copy()
    pool.query(np.array([0, 1, 2]))
    # The newly appended labeled samples should match.
    np.testing.assert_array_equal(pool.x_labeled[-3:], queried_x)
    np.testing.assert_array_equal(pool.y_labeled[-3:], queried_y)


# ---------------------------------------------------------------------------
# n_labeled / n_unlabeled properties
# ---------------------------------------------------------------------------


def test_n_labeled_property(classification_data):
    x_train, y_train, x_test, y_test = classification_data
    pool = ActiveLearningPool.from_dataset(x_train, y_train, x_test, y_test, initial_size=50, seed=0)
    assert pool.n_labeled == len(pool.x_labeled)


def test_n_unlabeled_property(classification_data):
    x_train, y_train, x_test, y_test = classification_data
    pool = ActiveLearningPool.from_dataset(x_train, y_train, x_test, y_test, initial_size=50, seed=0)
    assert pool.n_unlabeled == len(pool.x_unlabeled)
