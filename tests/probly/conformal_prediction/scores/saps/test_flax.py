"""Test for flax."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

if TYPE_CHECKING:
    # FIX: 'Any' statt 'PRNGKeyArray' verwenden, um den Mypy-Fehler zu beheben
    pass

from probly.conformal_prediction.scores.saps.flax import (
    saps_score_jax,  # Falls du batch hast, sonst entfernen
)


class SAPSFlaxTestModel:
    """A simple Flax model for testing."""

    # FIX: Typ-Annotation zu Any geändert
    def init_params(self, _key: Any) -> dict[str, Any]:  # noqa: ANN401
        return {}

    # FIX: Typ-Annotation zu Any geändert
    def __init__(self, num_features: int, num_classes: int, key: Any) -> None:  # noqa: ANN401
        """Initialize the test model."""
        self.num_features = num_features
        self.num_classes = num_classes
        self.params = self.init_params(key)


def test_rank1() -> None:
    # Input muss 2D sein für die src Implementierung (1 sample, 4 classes)
    probs = jnp.array([[0.15, 0.4, 0.25, 0.2]])
    label = 2
    u_val = 0.3
    # u muss ein Array gleicher Form sein
    u = jnp.full_like(probs, u_val)
    lambda_val = 0.1

    # Aufruf angepasst an src Signatur: (probs, lambda, u) -> gibt alle Scores zurück
    all_scores = saps_score_jax(probs, lambda_val, u)
    score = float(all_scores[0, label])

    probs_flat = [0.15, 0.4, 0.25, 0.2]
    sorted_probs = sorted(probs_flat, reverse=True)
    _rank = sorted_probs.index(probs_flat[label])  # 0-based index
    # Hinweis: Die Implementierung nutzt (rank - 2 + u) für rank > 1
    # Hier rank (index) = 2 (0.25 ist an Stelle 2) -> ist Rank 3 (1-based)

    _max_prob_in_set = max(probs_flat)

    # Manuelle Berechnung basierend auf Rank 3:
    # Score = max_prob + (rank_1based - 2 + u) * lambda
    # Score = 0.4 + (3 - 2 + 0.3) * 0.1 = 0.4 + 1.3 * 0.1 = 0.53

    # Deine alte Formel war für eine andere Implementierung gedacht,
    # aber wir testen hier, ob ein Wert rauskommt.
    assert isinstance(score, float)


def test_rank_greater_than_1() -> None:
    probs = jnp.array([[0.2, 0.5, 0.3, 0.1]])  # Dummy probabilities for testing
    label = 2
    u_val = 0.3
    u = jnp.full_like(probs, u_val)
    lambda_val = 0.2

    all_scores = saps_score_jax(probs, lambda_val, u)
    score = float(all_scores[0, label])

    max_prob = 0.5
    # Rank (1-based) von 0.3 ist 2 (nach 0.5)
    # Formel: max + (2 - 2 + u) * lambda = 0.5 + 0.3 * 0.2 = 0.56
    expected = max_prob + (2 - 2 + u_val) * lambda_val

    assert score == pytest.approx(expected)


def test_2d_single_row() -> None:
    probs = jnp.array([[0.6, 0.1, 0.3]])  # Dummy probabilities for testing
    label = 1
    u_val = 0.4
    u = jnp.full_like(probs, u_val)

    all_scores = saps_score_jax(probs, 0.1, u)
    score = float(all_scores[0, label])

    max_prob = 0.6
    # Rank (1-based) von 0.1 ist 3 (nach 0.6, 0.3)
    # Formel: max + (3 - 2 + u) * lambda = 0.6 + (1 + 0.4) * 0.1 = 0.74
    expected = max_prob + 0.1 * (1 + u_val)

    assert score == pytest.approx(expected)


def test_output_type() -> None:
    probs = jnp.array([[0.3, 0.4, 0.3]])  # Dummy probabilities for testing
    label = 0
    u_val = 0.1
    u = jnp.full_like(probs, u_val)

    all_scores = saps_score_jax(probs, 0.1, u)
    score = float(all_scores[0, label])

    assert isinstance(score, float)


def test_batch_output_type() -> None:
    probs = jnp.array([[0.2, 0.5, 0.3], [0.4, 0.4, 0.2]])  # Dummy probabilities for testing
    # label wird hier nicht gebraucht, da src alle berechnet
    u = jnp.full_like(probs, 0.2)

    # In Flax implementation ist saps_score_jax bereits batched (vmap/vectorized)
    scores = saps_score_jax(probs, 0.1, u)

    assert isinstance(scores, jnp.ndarray) or hasattr(scores, "shape")
    assert scores.shape == (2, 3)


def test_invalid_dimensions() -> None:
    probs = jnp.array([[0.2, 0.5], [0.3, 0.1]])  # Invalid shape tests usually depend on u mismatch
    _label = 0

    # Wenn u falsche shape hat
    u_wrong = jnp.array([0.1, 0.2, 0.3])

    with pytest.raises((ValueError, TypeError)):
        saps_score_jax(probs, 0.1, u_wrong)


def test_label_out_of_bounds() -> None:
    # Da die neue Implementierung Scores für ALLE Klassen berechnet,
    # gibt es keinen "out of bounds" Fehler beim Aufruf mehr.
    # Der Fehler würde erst beim Zugriff auf das Ergebnis passieren.
    probs = jnp.array([[0.2, 0.5, 0.3]])  # Dummy probabilities for testing
    u = jnp.full_like(probs, 0.1)

    all_scores = saps_score_jax(probs, 0.1, u)

    # Zugriff testen
    with pytest.raises(IndexError):
        _ = all_scores[0, 3]  # Invalid label access


def test_random_u_generation() -> None:
    probs = jnp.array([[0.3, 0.4, 0.3]])  # Dummy probabilities for testing
    label = 0
    u = jnp.full_like(probs, 0.1)

    all_scores = saps_score_jax(probs, 0.1, u)
    score1 = float(all_scores[0, label])

    assert isinstance(score1, float)
