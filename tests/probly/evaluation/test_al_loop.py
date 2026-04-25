"""Tests for the active_learning_steps iterator."""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from sklearn.linear_model import LogisticRegression  # noqa: E402

from probly.evaluation.active_learning.loop import ALState, active_learning_steps  # noqa: E402
from probly.evaluation.active_learning.pool import ActiveLearningPool  # noqa: E402
from probly.evaluation.active_learning.strategies import RandomQuery  # noqa: E402

# ---------------------------------------------------------------------------
# Test helper estimator
# ---------------------------------------------------------------------------


class _SklearnEstimator:
    """Thin wrapper around an sklearn model for testing the AL iterator."""

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
def estimator():
    """Unfitted _SklearnEstimator (the loop does the first fit)."""
    return _SklearnEstimator(LogisticRegression(max_iter=200))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_yields_initial_state(pool, estimator):
    """n_iterations=0 must yield exactly one state with iteration=0."""
    states = list(active_learning_steps(pool, estimator, RandomQuery(seed=1), n_iterations=0))
    assert len(states) == 1
    assert states[0].iteration == 0


def test_yields_correct_number(pool, estimator):
    """n_iterations=3 must yield 4 states: initial + 3 query-retrain cycles."""
    states = list(active_learning_steps(pool, estimator, RandomQuery(seed=1), query_size=10, n_iterations=3))
    assert len(states) == 4


def test_sequential_iteration_numbers(pool, estimator):
    """Iteration numbers must be 0, 1, 2, 3 in order."""
    states = list(active_learning_steps(pool, estimator, RandomQuery(seed=1), query_size=10, n_iterations=3))
    assert [s.iteration for s in states] == [0, 1, 2, 3]


def test_pool_grows_each_iteration(pool, estimator):
    """Labeled set must grow by query_size after each query-retrain step.

    Because the pool is mutated in place, we capture n_labeled at each yield
    rather than after exhausting the iterator.
    """
    query_size = 10
    labeled_sizes = []
    for state in active_learning_steps(pool, estimator, RandomQuery(seed=1), query_size=query_size, n_iterations=3):
        labeled_sizes.append(state.pool.n_labeled)  # noqa: PERF401

    initial_size = labeled_sizes[0]
    for i, size in enumerate(labeled_sizes):
        assert size == initial_size + i * query_size


def test_estimator_can_predict_after_training(pool, estimator):
    """Each yielded state's estimator must be able to predict on test data."""
    states = list(active_learning_steps(pool, estimator, RandomQuery(seed=1), query_size=10, n_iterations=2))
    x_test = pool.x_test
    for state in states:
        preds = state.estimator.predict(x_test)
        assert preds.shape == (len(x_test),)


def test_stops_when_pool_exhausted(classification_data):
    """Iterator must stop early when the unlabeled pool is emptied.

    With initial_size=130 from 150 training samples, unlabeled pool has 20
    samples. query_size=10, n_iterations=5 should yield 3 states:
    initial + 2 iterations (exhausting all 20 unlabeled samples).
    """
    x_train, y_train, x_test, y_test = classification_data
    small_pool = ActiveLearningPool.from_dataset(x_train, y_train, x_test, y_test, initial_size=130, seed=0)
    est = _SklearnEstimator(LogisticRegression(max_iter=200))
    states = list(active_learning_steps(small_pool, est, RandomQuery(seed=1), query_size=10, n_iterations=5))
    assert len(states) == 3


def test_alstate_is_dataclass():
    """ALState must be a dataclass with iteration, pool, and estimator fields."""
    assert dataclasses.is_dataclass(ALState)
    field_names = {f.name for f in dataclasses.fields(ALState)}
    assert field_names == {"iteration", "pool", "estimator"}
