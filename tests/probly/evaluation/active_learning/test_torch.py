"""PyTorch backend tests for active learning."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sklearn.linear_model import LogisticRegression  # noqa: E402

from probly.evaluation.active_learning.loop import active_learning_steps  # noqa: E402
from probly.evaluation.active_learning.pool import from_dataset  # noqa: E402
from probly.evaluation.active_learning.strategies import RandomQuery  # noqa: E402

from ._metrics_suite import MetricsSuite  # noqa: E402
from ._pool_suite import PoolSuite  # noqa: E402
from ._strategies_suite import StrategySuite  # noqa: E402

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
def arrays_equal():
    return torch.equal


@pytest.fixture
def concat_fn():
    return lambda arrs: torch.cat(arrs, dim=0)


@pytest.fixture
def sort_fn():
    def _sort(a):
        idx = torch.argsort(a[:, 0])
        return a[idx]

    return _sort


@pytest.fixture
def copy_fn():
    return lambda a: a.clone()


@pytest.fixture
def array_fn():
    return torch.tensor


@pytest.fixture
def make_one_hot_probs():
    def _make(y_true, n_classes):
        probs = torch.zeros(len(y_true), n_classes, dtype=torch.float32)
        probs[torch.arange(len(y_true)), y_true] = 1.0
        return probs

    return _make


@pytest.fixture
def make_random_probs():
    def _make(n, n_classes, seed):
        g = torch.Generator().manual_seed(seed)
        y_true = torch.randint(0, n_classes, (n,), generator=g)
        raw = torch.randn(n, n_classes, generator=g).exp()
        probs = raw / raw.sum(dim=1, keepdim=True)
        return probs, y_true

    return _make


# ---------------------------------------------------------------------------
# Estimator helper
# ---------------------------------------------------------------------------


class _TorchEstimator:
    """Thin sklearn wrapper returning torch tensors for testing."""

    def __init__(self) -> None:
        self._model = LogisticRegression(max_iter=200)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self._model.fit(x.numpy(), y.numpy())

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self._model.predict(x.numpy()))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self._model.predict_proba(x.numpy()).copy()).float()

    def uncertainty_scores(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x)
        return 1.0 - probs.max(dim=1).values


@pytest.fixture
def make_estimator():
    return _TorchEstimator


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestPool(PoolSuite):
    pass


class TestStrategies(StrategySuite):
    pass


class TestMetrics(MetricsSuite):
    pass


# ---------------------------------------------------------------------------
# Loop integration test
# ---------------------------------------------------------------------------


def test_loop_torch_end_to_end(classification_data, make_estimator):
    """Full AL loop with torch tensors."""
    x_train, y_train, x_test, y_test = classification_data
    pool = from_dataset(x_train, y_train, x_test, y_test, initial_size=50, seed=0)
    est = make_estimator()
    states = list(active_learning_steps(pool, est, RandomQuery(seed=1), query_size=10, n_iterations=3))
    assert len(states) == 4
    assert [s.iteration for s in states] == [0, 1, 2, 3]
