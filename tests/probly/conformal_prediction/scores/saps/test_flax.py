"""Test for flax."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("jax")

import jax.numpy as jnp
import jax.random as jrandom

from probly.conformal_prediction.scores.saps.flax import (
    saps_score_jax,
    saps_score_jax_batch,
)


class SAPSFlaxTestModel:
    """A simple Flax model for testing."""

    def init_params(self, _key: jrandom.PRNGKeyArray) -> dict[str, Any]:
        return {}

    def __init__(self, num_features: int, num_classes: int, key: jrandom.PRNGKeyArray) -> None:
        """Initialize the test model."""
        self.num_features = num_features
        self.num_classes = num_classes
        self.params = self.init_params(key)


def test_rank1() -> None:
    probs = jnp.array([0.15, 0.4, 0.25, 0.2])
    label = 2
    u = 0.3
    lambda_val = 0.1

    score = saps_score_jax(probs, label, lambda_val, u)

    sorted_probs = sorted(probs, reverse=True)
    rank = sorted_probs.index(probs[label])
    max_prob_in_set = max(probs)
    expected = float(max_prob_in_set + lambda_val * (rank + u))

    assert score == pytest.approx(expected)


def test_rank_greater_than_1() -> None:
    probs = jnp.array([0.2, 0.5, 0.3, 0.1])  # Dummy probabilities for testing
    label = 2
    u = 0.3
    lambda_val = 0.2

    score = saps_score_jax(probs, label, lambda_val, u)

    max_prob = 0.5
    expected = max_prob + lambda_val * (1 + u)
    assert score == expected


def test_2d_single_row() -> None:
    probs = jnp.array([[0.6, 0.1, 0.3]])  # Dummy probabilities for testing
    label = 1
    u = 0.4

    score = saps_score_jax(probs, label, lambda_val=0.1, u=u)

    max_prob = 0.6
    expected = max_prob + 0.1 * (1 + u)
    assert score == pytest.approx(expected)


def test_output_type() -> None:
    probs = jnp.array([0.3, 0.4, 0.3])  # Dummy probabilities for testing
    label = 0
    u = 0.1

    score = saps_score_jax(probs, label, lambda_val=0.1, u=u)

    assert isinstance(score, float)


def test_batch_output_type() -> None:
    probs = jnp.array([[0.2, 0.5, 0.3], [0.4, 0.4, 0.2]])  # Dummy probabilities for testing
    label = jnp.array([1, 0])
    u = jnp.array([0.2, 0.4])

    scores = saps_score_jax_batch(probs, label, lambda_val=0.1, us=u)
    assert isinstance(scores, jnp.ndarray)
    assert scores.shape == (2,)


def test_invalid_dimensions() -> None:
    probs = jnp.array([[0.2, 0.5], [0.3, 0.1]])  # Invalid shape
    label = 0

    with pytest.raises(ValueError):  # noqa: PT011
        saps_score_jax(probs, label, lambda_val=0.1)


def test_label_out_of_bounds() -> None:
    probs = jnp.array([0.2, 0.5, 0.3])  # Dummy probabilities for testing
    label = 3  # Invalid label

    with pytest.raises(ValueError):  # noqa: PT011
        saps_score_jax(probs, label, lambda_val=0.1)


def test_random_u_generation() -> None:
    probs = jnp.array([0.3, 0.4, 0.3])  # Dummy probabilities for testing
    label = 0

    score1 = saps_score_jax(probs, label, lambda_val=0.1)

    assert isinstance(score1, float)
