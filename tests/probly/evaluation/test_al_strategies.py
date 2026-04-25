"""Tests for active learning query strategies."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sklearn.linear_model import LogisticRegression  # noqa: E402

from probly.evaluation.active_learning.pool import ActiveLearningPool  # noqa: E402
from probly.evaluation.active_learning.strategies import (  # noqa: E402
    BADGEQuery,
    EntropyQuery,
    MarginSampling,
    RandomQuery,
    UncertaintyQuery,
)

# ---------------------------------------------------------------------------
# Test helper estimator
# ---------------------------------------------------------------------------


class _SklearnEstimator:
    """Thin wrapper around an sklearn model for testing query strategies."""

    def __init__(self, model) -> None:
        self._model = model

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self._model.fit(x.numpy(), y.numpy())

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self._model.predict(x.numpy()))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self._model.predict_proba(x.numpy()).copy()).float()

    def uncertainty_scores(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x)
        return 1.0 - probs.max(dim=1).values


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classification_data():
    """200 samples, 5 features, 3 classes, 150/50 train/test split."""
    g = torch.Generator().manual_seed(42)
    n_total = 200
    n_features = 5
    n_classes = 3
    x = torch.randn(n_total, n_features, generator=g)
    y = torch.randint(0, n_classes, (n_total,), generator=g)
    x_train, y_train = x[:150], y[:150]
    x_test, y_test = x[150:], y[150:]
    return x_train, y_train, x_test, y_test


@pytest.fixture
def pool(classification_data):
    """ActiveLearningPool with 50 labeled, 100 unlabeled samples."""
    x_train, y_train, x_test, y_test = classification_data
    return ActiveLearningPool.from_dataset(x_train, y_train, x_test, y_test, initial_size=50, seed=0)


@pytest.fixture
def estimator(pool):
    """Fitted _SklearnEstimator for use in strategy tests."""
    est = _SklearnEstimator(LogisticRegression(max_iter=200))
    est.fit(pool.x_labeled, pool.y_labeled)
    return est


# ---------------------------------------------------------------------------
# Parametrized tests: basic contracts for prob-based strategies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy_cls",
    [RandomQuery, MarginSampling, EntropyQuery, UncertaintyQuery],
)
def test_select_returns_correct_count(strategy_cls, estimator, pool):
    """select() must return exactly n indices."""
    strategy = strategy_cls()
    indices = strategy.select(estimator, pool, n=10)
    assert len(indices) == 10


@pytest.mark.parametrize(
    "strategy_cls",
    [RandomQuery, MarginSampling, EntropyQuery, UncertaintyQuery],
)
def test_select_returns_valid_indices(strategy_cls, estimator, pool):
    """All returned indices must be valid positions into pool.x_unlabeled."""
    strategy = strategy_cls()
    indices = strategy.select(estimator, pool, n=10)
    assert np.all(indices >= 0)
    assert np.all(indices < pool.n_unlabeled)


@pytest.mark.parametrize(
    "strategy_cls",
    [RandomQuery, MarginSampling, EntropyQuery, UncertaintyQuery],
)
def test_select_returns_unique_indices(strategy_cls, estimator, pool):
    """All returned indices must be distinct."""
    strategy = strategy_cls()
    indices = strategy.select(estimator, pool, n=10)
    assert len(indices) == len(np.unique(indices))


# ---------------------------------------------------------------------------
# RandomQuery specific tests
# ---------------------------------------------------------------------------


def test_random_query_varies_with_different_seeds(estimator, pool):
    """RandomQuery must produce different selections for different seeds."""
    indices_a = RandomQuery(seed=0).select(estimator, pool, n=10)
    indices_b = RandomQuery(seed=99).select(estimator, pool, n=10)
    assert not np.array_equal(np.sort(indices_a), np.sort(indices_b))


# ---------------------------------------------------------------------------
# MarginSampling quality test
# ---------------------------------------------------------------------------


def test_margin_sampling_selects_uncertain_samples(estimator, pool):
    """MarginSampling should select samples with smaller margin than average."""
    n = 10
    strategy = MarginSampling()
    indices = strategy.select(estimator, pool, n=n)

    probs = estimator.predict_proba(pool.x_unlabeled)
    sorted_probs = probs.sort(dim=1).values
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]

    selected_margin = margins[indices].mean().item()
    mask = torch.ones(len(margins), dtype=torch.bool)
    mask[indices] = False
    remaining_margin = margins[mask].mean().item()

    assert selected_margin <= remaining_margin


# ---------------------------------------------------------------------------
# BADGEQuery tests
# ---------------------------------------------------------------------------


def test_badge_query_correct_count(estimator, pool):
    """BADGEQuery must return exactly n indices."""
    strategy = BADGEQuery()
    indices = strategy.select(estimator, pool, n=10)
    assert len(indices) == 10


def test_badge_query_unique_indices(estimator, pool):
    """BADGEQuery must return distinct indices."""
    strategy = BADGEQuery()
    indices = strategy.select(estimator, pool, n=10)
    assert len(indices) == len(np.unique(indices))


# ---------------------------------------------------------------------------
# Clamping test: requesting more than pool size
# ---------------------------------------------------------------------------


def test_clamping_returns_pool_size(estimator, pool):
    """When n > pool.n_unlabeled, select() should return pool.n_unlabeled items."""
    strategy = RandomQuery(seed=0)
    indices = strategy.select(estimator, pool, n=pool.n_unlabeled + 100)
    assert len(indices) == pool.n_unlabeled
