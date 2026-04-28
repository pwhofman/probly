"""Shared strategy test suite for all backends."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from probly.evaluation.active_learning.pool import from_dataset
from probly.evaluation.active_learning.strategies import (
    BADGEQuery,
    EntropySampling,
    LeastConfidentSampling,
    MarginSampling,
    RandomQuery,
    UncertaintyQuery,
)


class StrategySuite:
    """Backend-agnostic strategy tests.

    Requires fixtures: classification_data, make_estimator, to_numpy, margin_fn.
    """

    @pytest.fixture
    def pool(self, classification_data):
        x_train, y_train, x_test, y_test = classification_data
        return from_dataset(x_train, y_train, x_test, y_test, initial_size=50, seed=0)

    @pytest.fixture
    def estimator(self, pool, make_estimator):
        est = make_estimator()
        est.fit(pool.x_labeled, pool.y_labeled)
        return est

    @pytest.mark.parametrize(
        "strategy_cls", [RandomQuery, EntropySampling, LeastConfidentSampling, MarginSampling, UncertaintyQuery]
    )
    def test_select_returns_correct_count(self, strategy_cls, estimator, pool, to_numpy):
        strategy = strategy_cls()
        indices = to_numpy(strategy.select(estimator, pool, n=10))
        assert len(indices) == 10

    @pytest.mark.parametrize(
        "strategy_cls", [RandomQuery, EntropySampling, LeastConfidentSampling, MarginSampling, UncertaintyQuery]
    )
    def test_select_returns_valid_indices(self, strategy_cls, estimator, pool, to_numpy):
        strategy = strategy_cls()
        indices = to_numpy(strategy.select(estimator, pool, n=10))
        assert np.all(indices >= 0)
        assert np.all(indices < pool.n_unlabeled)

    @pytest.mark.parametrize(
        "strategy_cls", [RandomQuery, EntropySampling, LeastConfidentSampling, MarginSampling, UncertaintyQuery]
    )
    def test_select_returns_unique_indices(self, strategy_cls, estimator, pool, to_numpy):
        strategy = strategy_cls()
        indices = to_numpy(strategy.select(estimator, pool, n=10))
        assert len(indices) == len(np.unique(indices))

    def test_random_query_varies_with_different_seeds(self, estimator, pool, to_numpy):
        indices_a = to_numpy(RandomQuery(seed=0).select(estimator, pool, n=10))
        indices_b = to_numpy(RandomQuery(seed=99).select(estimator, pool, n=10))
        assert not np.array_equal(np.sort(indices_a), np.sort(indices_b))

    def test_entropy_selects_uncertain_samples(self, estimator, pool, to_numpy):
        """EntropySampling should select samples with higher entropy than average."""
        n = 10
        strategy = EntropySampling()
        indices = to_numpy(strategy.select(estimator, pool, n=n))
        probs = to_numpy(estimator.predict_proba(pool.x_unlabeled))
        h = -np.sum(probs * np.log(np.clip(probs, 1e-12, None)), axis=1)
        selected_h = float(np.mean(h[indices]))
        remaining_mask = np.ones(len(h), dtype=bool)
        remaining_mask[indices] = False
        remaining_h = float(np.mean(h[remaining_mask]))
        assert selected_h >= remaining_h

    def test_least_confident_selects_uncertain_samples(self, estimator, pool, to_numpy):
        """LeastConfidentSampling should select samples with lower max-prob than average."""
        n = 10
        strategy = LeastConfidentSampling()
        indices = to_numpy(strategy.select(estimator, pool, n=n))
        probs = estimator.predict_proba(pool.x_unlabeled)
        confidence = to_numpy(probs).max(axis=1)
        selected_conf = float(np.mean(confidence[indices]))
        remaining_mask = np.ones(len(confidence), dtype=bool)
        remaining_mask[indices] = False
        remaining_conf = float(np.mean(confidence[remaining_mask]))
        assert selected_conf <= remaining_conf

    def test_margin_sampling_selects_uncertain_samples(self, estimator, pool, margin_fn, to_numpy):
        """MarginSampling should select samples with smaller margin than average."""
        n = 10
        strategy = MarginSampling()
        indices = to_numpy(strategy.select(estimator, pool, n=n))
        probs = estimator.predict_proba(pool.x_unlabeled)
        margins = margin_fn(probs)
        selected_margin = float(np.mean(margins[indices]))
        remaining_mask = np.ones(len(margins), dtype=bool)
        remaining_mask[indices] = False
        remaining_margin = float(np.mean(margins[remaining_mask]))
        assert selected_margin <= remaining_margin

    def test_badge_query_correct_count(self, estimator, pool, to_numpy):
        strategy = BADGEQuery()
        with pytest.warns(UserWarning, match="does not implement BadgeEstimator"):
            indices = to_numpy(strategy.select(estimator, pool, n=10))
        assert len(indices) == 10

    def test_badge_query_unique_indices(self, estimator, pool, to_numpy):
        strategy = BADGEQuery()
        with pytest.warns(UserWarning, match="does not implement BadgeEstimator"):
            indices = to_numpy(strategy.select(estimator, pool, n=10))
        assert len(indices) == len(np.unique(indices))

    def test_badge_query_uses_embeddings(self, pool, make_badge_estimator, to_numpy):
        """BADGEQuery uses embed() when estimator implements BadgeEstimator."""
        est = make_badge_estimator()
        est.fit(pool.x_labeled, pool.y_labeled)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # fail if warning is raised
            strategy = BADGEQuery()
            indices = to_numpy(strategy.select(est, pool, n=10))
        assert len(indices) == 10
        assert len(indices) == len(np.unique(indices))

    def test_uncertainty_query_selects_uncertain_samples(self, pool, make_estimator, to_numpy):
        """UncertaintyQuery should select samples with higher uncertainty than average."""
        est = make_estimator()
        est.fit(pool.x_labeled, pool.y_labeled)
        n = 10
        strategy = UncertaintyQuery()
        indices = to_numpy(strategy.select(est, pool, n=n))
        scores = to_numpy(est.uncertainty_scores(pool.x_unlabeled))
        selected_score = float(np.mean(scores[indices]))
        remaining_mask = np.ones(len(scores), dtype=bool)
        remaining_mask[indices] = False
        remaining_score = float(np.mean(scores[remaining_mask]))
        assert selected_score >= remaining_score

    def test_select_with_zero_n(self, estimator, pool, to_numpy):
        """Requesting n=0 samples should return an empty array."""
        strategy = RandomQuery(seed=0)
        indices = to_numpy(strategy.select(estimator, pool, n=0))
        assert len(indices) == 0

    def test_clamping_returns_pool_size(self, estimator, pool, to_numpy):
        strategy = RandomQuery(seed=0)
        indices = to_numpy(strategy.select(estimator, pool, n=pool.n_unlabeled + 100))
        assert len(indices) == pool.n_unlabeled
