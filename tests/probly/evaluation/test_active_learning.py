"""Tests for the active_learning evaluation module."""
# ruff: noqa: N806

from __future__ import annotations

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn.datasets import make_classification, make_regression  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

from probly.evaluation.active_learning import active_learning_loop  # noqa: E402
from probly.evaluation.active_learning._utils import compute_normalized_auc, resolve_metric  # noqa: E402

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classification_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)
    return train_test_split(X, y, test_size=0.3, random_state=0)


@pytest.fixture
def regression_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=0)
    return train_test_split(X, y, test_size=0.3, random_state=0)


@pytest.fixture
def clf() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, random_state=0)


@pytest.fixture
def reg() -> LinearRegression:
    return LinearRegression()


# ---------------------------------------------------------------------------
# Return type and basic shapes
# ---------------------------------------------------------------------------


def test_returns_tuple_of_two(classification_data, clf):
    X_train, X_test, y_train, y_test = classification_data
    result = active_learning_loop(clf, X_train, y_train, X_test, y_test, pool_size=5, n_iterations=3, seed=0)
    assert isinstance(result, tuple)
    assert len(result) == 4


def test_scores_length_matches_iterations(classification_data, clf):
    x_train, x_test, y_train, y_test = classification_data
    _, _, scores, _ = active_learning_loop(clf, x_train, y_train, x_test, y_test, pool_size=5, n_iterations=3, seed=0)
    assert len(scores) == 3


def test_normalized_auc_is_float(classification_data, clf):
    x_train, x_test, y_train, y_test = classification_data
    _, _, _, nauc = active_learning_loop(clf, x_train, y_train, x_test, y_test, pool_size=5, n_iterations=3, seed=0)
    assert isinstance(nauc, float)
    assert 0.0 <= nauc <= 1.0


# ---------------------------------------------------------------------------
# Metric string variants
# ---------------------------------------------------------------------------


def test_metric_accuracy(classification_data, clf):
    X_train, X_test, y_train, y_test = classification_data
    _, _, scores, _ = active_learning_loop(
        clf, X_train, y_train, X_test, y_test, metric="accuracy", pool_size=5, n_iterations=3, seed=0
    )
    for s in scores:
        assert 0.0 <= s <= 1.0


def test_metric_mse_is_negative(regression_data, reg):
    X_train, X_test, y_train, y_test = regression_data
    _, _, scores, _ = active_learning_loop(
        reg, X_train, y_train, X_test, y_test, metric="mse", pool_size=5, n_iterations=3, seed=0
    )
    for s in scores:
        assert s <= 0.0


def test_metric_mae_is_negative(regression_data, reg):
    X_train, X_test, y_train, y_test = regression_data
    _, _, scores, _ = active_learning_loop(
        reg, X_train, y_train, X_test, y_test, metric="mae", pool_size=5, n_iterations=3, seed=0
    )
    for s in scores:
        assert s <= 0.0


def test_metric_callable(classification_data, clf):
    X_train, X_test, y_train, y_test = classification_data

    def custom_metric(y_true, y_pred):
        return float(np.mean(y_pred == y_true))

    _, _, scores, _ = active_learning_loop(
        clf, X_train, y_train, X_test, y_test, metric=custom_metric, pool_size=5, n_iterations=3, seed=0
    )
    assert len(scores) == 3


def test_metric_unknown_string_raises(classification_data, clf):
    X_train, X_test, y_train, y_test = classification_data
    with pytest.raises(ValueError, match="Unknown metric"):
        active_learning_loop(clf, X_train, y_train, X_test, y_test, metric="f1", pool_size=5, n_iterations=1)


# ---------------------------------------------------------------------------
# Seed reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_same_result(classification_data, clf):
    X_train, X_test, y_train, y_test = classification_data
    _, _, s1, nauc1 = active_learning_loop(clf, X_train, y_train, X_test, y_test, pool_size=5, n_iterations=3, seed=42)
    _, _, s2, nauc2 = active_learning_loop(clf, X_train, y_train, X_test, y_test, pool_size=5, n_iterations=3, seed=42)
    assert s1 == s2
    assert nauc1 == nauc2


def test_different_seeds_may_differ(classification_data, clf):
    X_train, X_test, y_train, y_test = classification_data
    _, _, s1, _ = active_learning_loop(clf, X_train, y_train, X_test, y_test, pool_size=5, n_iterations=2, seed=0)
    _, _, s2, _ = active_learning_loop(clf, X_train, y_train, X_test, y_test, pool_size=5, n_iterations=2, seed=99)
    assert s1 != s2


# ---------------------------------------------------------------------------
# Pool exhaustion
# ---------------------------------------------------------------------------


def test_stops_early_when_pool_exhausted(clf):
    X, y = make_classification(n_samples=20, n_features=4, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # pool_size=10, only ~14 train samples → exhausted after first or second query
    _, _, scores, _ = active_learning_loop(clf, X_train, y_train, X_test, y_test, pool_size=10, n_iterations=20, seed=0)
    assert len(scores) < 20


# ---------------------------------------------------------------------------
# Custom query function
# ---------------------------------------------------------------------------


def test_custom_query_fn_is_called(classification_data, clf):
    X_train, X_test, y_train, y_test = classification_data
    call_count = {"n": 0}

    def counting_query_fn(outputs):
        call_count["n"] += 1
        return np.random.default_rng(0).random(len(outputs))

    active_learning_loop(
        clf, X_train, y_train, X_test, y_test, query_fn=counting_query_fn, pool_size=5, n_iterations=3, seed=0
    )
    assert call_count["n"] == 3


# ---------------------------------------------------------------------------
# Regression (no task param needed)
# ---------------------------------------------------------------------------


def test_regression_scores_are_negative_mse(regression_data, reg):
    X_train, X_test, y_train, y_test = regression_data
    _, _, scores, _ = active_learning_loop(reg, X_train, y_train, X_test, y_test, pool_size=5, n_iterations=3, seed=0)
    for s in scores:
        assert s <= 0.0


# ---------------------------------------------------------------------------
# Stochastic sampling (num_samples > 1)
# ---------------------------------------------------------------------------


def test_num_samples_greater_one_classification(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    clf = MLPClassifier(hidden_layer_sizes=(8,), max_iter=200, random_state=0)
    _, _, scores, nauc = active_learning_loop(
        clf, X_train, y_train, X_test, y_test, pool_size=5, n_iterations=2, num_samples=5, seed=0
    )
    assert len(scores) == 2
    assert isinstance(nauc, float)


# ---------------------------------------------------------------------------
# Torch tensor input
# ---------------------------------------------------------------------------


def test_torch_tensor_input(clf):
    torch = pytest.importorskip("torch")
    X, y = make_classification(n_samples=100, n_features=10, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    _, _, scores, _ = active_learning_loop(
        clf, X_train_t, y_train_t, X_test_t, y_test_t, pool_size=5, n_iterations=2, seed=0
    )
    assert len(scores) == 2


# ---------------------------------------------------------------------------
# compute_normalized_auc unit tests
# ---------------------------------------------------------------------------


def test_nauc_constant_one_is_one():
    # Ideal strategy: always scores 1.0
    assert compute_normalized_auc([1.0, 1.0, 1.0, 1.0]) == 1.0


def test_nauc_constant_below_one():
    # Constant score of 0.8 → NAUC = 0.8
    assert compute_normalized_auc([0.8, 0.8, 0.8, 0.8]) == pytest.approx(0.8)


def test_nauc_range():
    nauc = compute_normalized_auc([0.5, 0.6, 0.7, 0.8])
    assert 0.0 <= nauc <= 1.0


def test_nauc_slow_improver_lower_than_fast():
    slow = compute_normalized_auc([0.5, 0.5, 0.5, 0.8])
    fast = compute_normalized_auc([0.5, 0.6, 0.7, 0.8])
    assert slow < fast


def test_nauc_trapezoid_rule():
    # trapz([0, 1], x=[0, 1]) = 0.5  →  NAUC = 0.5 / 1 = 0.5
    assert compute_normalized_auc([0.0, 1.0]) == pytest.approx(0.5)


def test_nauc_nan_for_single_value():
    assert np.isnan(compute_normalized_auc([0.5]))


# ---------------------------------------------------------------------------
# resolve_metric unit tests
# ---------------------------------------------------------------------------


def test_resolve_metric_mse_negated():
    fn, name = resolve_metric("mse")
    assert name == "mse"
    assert fn(np.array([0.0]), np.array([1.0])) < 0


def test_resolve_metric_mae_negated():
    fn, name = resolve_metric("mae")
    assert name == "mae"
    assert fn(np.array([0.0]), np.array([1.0])) < 0


def test_resolve_metric_accuracy():
    fn, name = resolve_metric("accuracy")
    assert name == "accuracy"
    assert fn(np.array([1, 0, 1]), np.array([1, 0, 1])) == 1.0


def test_resolve_metric_callable_passthrough():
    def custom(_yt, _yp):
        return 42.0

    fn, name = resolve_metric(custom)
    assert name is None
    assert fn(None, None) == 42.0


def test_resolve_metric_none_defaults_to_neg_mse():
    fn, name = resolve_metric(None)
    assert name == "mse"
    assert fn(np.array([0.0]), np.array([1.0])) < 0
