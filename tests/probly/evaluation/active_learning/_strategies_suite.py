"""Shared strategy test suite for all backends."""

from __future__ import annotations

import numpy as np
import pytest

from probly.evaluation.active_learning.pool import from_dataset
from probly.evaluation.active_learning.strategies import (
    BADGEQuery,
    MarginSampling,
    RandomQuery,
    UncertaintyQuery,
)


class StrategySuite:
    """Backend-agnostic strategy tests. Requires fixtures: classification_data, make_estimator."""

    @pytest.fixture
    def pool(self, classification_data):
        x_train, y_train, x_test, y_test = classification_data
        return from_dataset(x_train, y_train, x_test, y_test, initial_size=50, seed=0)

    @pytest.fixture
    def estimator(self, pool, make_estimator):
        est = make_estimator()
        est.fit(pool.x_labeled, pool.y_labeled)
        return est

    @pytest.mark.parametrize("strategy_cls", [RandomQuery, MarginSampling, UncertaintyQuery])
    def test_select_returns_correct_count(self, strategy_cls, estimator, pool):
        strategy = strategy_cls()
        indices = strategy.select(estimator, pool, n=10)
        assert len(indices) == 10

    @pytest.mark.parametrize("strategy_cls", [RandomQuery, MarginSampling, UncertaintyQuery])
    def test_select_returns_valid_indices(self, strategy_cls, estimator, pool):
        strategy = strategy_cls()
        indices = strategy.select(estimator, pool, n=10)
        assert np.all(indices >= 0)
        assert np.all(indices < pool.n_unlabeled)

    @pytest.mark.parametrize("strategy_cls", [RandomQuery, MarginSampling, UncertaintyQuery])
    def test_select_returns_unique_indices(self, strategy_cls, estimator, pool):
        strategy = strategy_cls()
        indices = strategy.select(estimator, pool, n=10)
        assert len(indices) == len(np.unique(indices))

    def test_random_query_varies_with_different_seeds(self, estimator, pool):
        indices_a = RandomQuery(seed=0).select(estimator, pool, n=10)
        indices_b = RandomQuery(seed=99).select(estimator, pool, n=10)
        assert not np.array_equal(np.sort(indices_a), np.sort(indices_b))

    def test_margin_sampling_selects_uncertain_samples(self, estimator, pool, margin_fn):
        """MarginSampling should select samples with smaller margin than average."""
        n = 10
        strategy = MarginSampling()
        indices = strategy.select(estimator, pool, n=n)
        probs = estimator.predict_proba(pool.x_unlabeled)
        margins = margin_fn(probs)
        selected_margin = float(np.mean(margins[indices]))
        remaining_mask = np.ones(len(margins), dtype=bool)
        remaining_mask[indices] = False
        remaining_margin = float(np.mean(margins[remaining_mask]))
        assert selected_margin <= remaining_margin

    def test_badge_query_correct_count(self, estimator, pool):
        strategy = BADGEQuery()
        indices = strategy.select(estimator, pool, n=10)
        assert len(indices) == 10

    def test_badge_query_unique_indices(self, estimator, pool):
        strategy = BADGEQuery()
        indices = strategy.select(estimator, pool, n=10)
        assert len(indices) == len(np.unique(indices))

    def test_clamping_returns_pool_size(self, estimator, pool):
        strategy = RandomQuery(seed=0)
        indices = strategy.select(estimator, pool, n=pool.n_unlabeled + 100)
        assert len(indices) == pool.n_unlabeled
