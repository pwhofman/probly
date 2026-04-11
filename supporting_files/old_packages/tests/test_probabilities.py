"""Tests for utils.probabilities functions."""

from __future__ import annotations

import numpy as np

from probly.utils.probabilities import differential_entropy_gaussian, intersection_probability, kl_divergence_gaussian


def test_differential_entropy_gaussian() -> None:
    assert np.isclose(differential_entropy_gaussian(0.5), 1.54709559)
    assert np.allclose(differential_entropy_gaussian(np.array([1, 2]), base=np.e), np.array([1.41893853, 1.76551212]))


def test_kl_divergence_gaussian() -> None:
    mu1 = np.array([0.0, 1.0])
    mu2 = np.array([1.0, 0.0])
    sigma21 = np.array([0.1, 0.1])
    sigma22 = np.array([0.1, 0.1])
    assert np.isclose(kl_divergence_gaussian(1.0, 1.0, 1.0, 1.0), 0.0)
    assert np.isclose(kl_divergence_gaussian(1.0, 1.0, 1.0, 1.0, base=np.e), 0.0)
    assert np.allclose(kl_divergence_gaussian(mu1, sigma21, mu2, sigma22, base=np.e), np.array([5.0, 5.0]))


def test_intersection_probability() -> None:
    rng = np.random.default_rng()

    probs = rng.dirichlet(np.ones(2), size=(5, 5))
    int_prob = intersection_probability(probs)
    assert int_prob.shape == (5, 2)
    assert np.allclose(np.sum(int_prob, axis=1), 1.0)

    probs = rng.dirichlet(np.ones(10), size=(5, 5))
    int_prob = intersection_probability(probs)
    assert int_prob.shape == (5, 10)
    assert np.allclose(np.sum(int_prob, axis=1), 1.0)

    probs = np.array([1 / 3, 1 / 3, 1 / 3] * 5).reshape(1, 5, 3)
    int_prob = intersection_probability(probs)
    assert int_prob.shape == (1, 3)
    assert np.allclose(np.sum(int_prob, axis=1), 1.0)
