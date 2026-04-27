"""NumPy backend tests for active learning."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from ._loop_suite import LoopSuite
from ._metrics_suite import MetricsSuite
from ._pool_suite import PoolSuite
from ._strategies_suite import StrategySuite

# ---------------------------------------------------------------------------
# Fixtures
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


@pytest.fixture
def arrays_equal():
    return np.array_equal


@pytest.fixture
def concat_fn():
    return lambda arrs: np.concatenate(arrs, axis=0)


@pytest.fixture
def sort_fn():
    return lambda a: a[np.lexsort(a.T)]


@pytest.fixture
def copy_fn():
    return lambda a: a.copy()


@pytest.fixture
def array_fn():
    return np.array


@pytest.fixture
def make_one_hot_probs():
    def _make(y_true, n_classes):
        probs = np.zeros((len(y_true), n_classes), dtype=np.float32)
        probs[np.arange(len(y_true)), y_true] = 1.0
        return probs

    return _make


@pytest.fixture
def make_random_probs():
    def _make(n, n_classes, seed):
        rng = np.random.default_rng(seed)
        y_true = rng.integers(0, n_classes, size=n)
        raw = np.exp(rng.standard_normal((n, n_classes)))
        probs = (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)
        return probs, y_true

    return _make


# ---------------------------------------------------------------------------
# Estimator helper
# ---------------------------------------------------------------------------


class _NumpyEstimator:
    """Thin sklearn wrapper returning numpy arrays for testing."""

    def __init__(self) -> None:
        self._model = LogisticRegression(max_iter=200)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(x).astype(np.float32)

    def uncertainty_scores(self, x: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(x)
        return 1.0 - probs.max(axis=1)


@pytest.fixture
def margin_fn():
    def _margin(probs):
        sorted_probs = np.sort(probs, axis=1)
        return sorted_probs[:, -1] - sorted_probs[:, -2]

    return _margin


@pytest.fixture
def make_estimator():
    return _NumpyEstimator


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestPool(PoolSuite):
    pass


class TestStrategies(StrategySuite):
    pass


class TestMetrics(MetricsSuite):
    pass


class TestLoop(LoopSuite):
    pass
