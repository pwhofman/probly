"""Tests for utils.numpy functions."""

from __future__ import annotations

import math

import numpy as np

from probly.utils import elementwise_entropy, entropy, machine_epsilon
from probly.utils.numpy import numpy_elementwise_entropy, numpy_entropy, numpy_machine_epsilon


def test_entropy_dispatches_to_numpy() -> None:
    p = np.array([[0.5, 0.5], [1.0, 0.0]])
    np.testing.assert_array_equal(entropy(p), numpy_entropy(p))
    np.testing.assert_allclose(entropy(p), [math.log(2.0), 0.0])


def test_elementwise_entropy_dispatches_to_numpy() -> None:
    p = np.array([0.5, 1.0, 0.0, -1e-9])
    np.testing.assert_array_equal(elementwise_entropy(p), numpy_elementwise_entropy(p))
    # Zero and negative entries contribute nothing.
    np.testing.assert_allclose(elementwise_entropy(p), [0.5 * math.log(2.0), 0.0, 0.0, 0.0])


def test_machine_epsilon_dispatches_to_numpy() -> None:
    single, double = np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float64)
    assert machine_epsilon(single) == numpy_machine_epsilon(single) == np.finfo(np.float32).eps
    assert machine_epsilon(double) == np.finfo(np.float64).eps
    assert isinstance(machine_epsilon(single), float)
