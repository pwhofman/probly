"""Fixtures for sklearn models."""

from __future__ import annotations

import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.base import BaseEstimator  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.neural_network import MLPClassifier, MLPRegressor  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # noqa: E402


@pytest.fixture
def sklearn_random_state() -> int:
    """Return a random state for sklearn models."""
    return 0


@pytest.fixture
def sklearn_logistic_regression(sklearn_random_state: int) -> BaseEstimator:
    """Return a small classification model with 2 input features."""
    model = LogisticRegression(random_state=sklearn_random_state)
    return model


@pytest.fixture
def sklearn_mlp_classifier_2d_2d(sklearn_random_state: int) -> BaseEstimator:
    """Return a small MLP classification model with 2 input and 2 output neurons."""
    model = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=1000, random_state=sklearn_random_state)
    return model


@pytest.fixture
def sklearn_mlp_regressor_2d_1d(sklearn_random_state: int) -> MLPRegressor:
    """Return a small MLP regression model with 2 input and 1 output neurons."""
    model = MLPRegressor(hidden_layer_sizes=(5, 5), max_iter=1000, random_state=sklearn_random_state)
    return model


@pytest.fixture
def sklearn_decision_tree_classifier(sklearn_random_state: int) -> BaseEstimator:
    """Return a small decision tree classification model."""
    model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1, random_state=sklearn_random_state)
    return model


@pytest.fixture
def sklearn_decision_tree_regressor(sklearn_random_state: int) -> BaseEstimator:
    """Return a small decision tree regression model."""
    model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=sklearn_random_state)
    return model
