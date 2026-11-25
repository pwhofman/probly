"""Tests for utils.numpy functions."""

from __future__ import annotations

import math

import numpy as np

from probly.utils import elementwise_entropy, entropy
from probly.utils.numpy import numpy_elementwise_entropy, numpy_entropy


def test_entropy_dispatches_to_numpy() -> None:
    p = np.array([[0.5, 0.5], [1.0, 0.0]])
    np.testing.assert_array_equal(entropy(p), numpy_entropy(p))
    np.testing.assert_allclose(entropy(p), [math.log(2.0), 0.0])


def test_elementwise_entropy_dispatches_to_numpy() -> None:
    p = np.array([0.5, 1.0, 0.0, -1e-9])
    np.testing.assert_array_equal(elementwise_entropy(p), numpy_elementwise_entropy(p))
    # Zero and negative entries contribute nothing.
    np.testing.assert_allclose(elementwise_entropy(p), [0.5 * math.log(2.0), 0.0, 0.0, 0.0])
