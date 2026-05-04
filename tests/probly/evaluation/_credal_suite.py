"""Backend-agnostic credal-set suite for the envelope-based metrics.

Only credal-set families that exist on every supported backend are
parametrised here. ``Singleton`` and ``Discrete`` credal sets currently exist
only as numpy types; their tests live in :mod:`tests.probly.evaluation.test_credal`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from probly.evaluation import average_interval_width, coverage, efficiency

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


class CredalSuite:
    """Tests that exercise the envelope-based credal-set handlers per backend."""

    def test_convex_uses_interval_dominance(
        self, make_convex: Callable[[np.ndarray], Any], array_fn: Callable[..., Any]
    ) -> None:
        """Coverage and efficiency on a single sample with two vertices."""
        probs = np.array([[[0.6, 0.3, 0.1], [0.4, 0.5, 0.1]]])
        cs = make_convex(probs)
        # lower = [0.4, 0.3, 0.1]; upper = [0.6, 0.5, 0.1]; max(lower) = 0.4.
        # mask = upper >= 0.4 = [True, True, False]; truth 1 covered.
        assert coverage(cs, array_fn([1])) == pytest.approx(1.0)
        assert efficiency(cs) == pytest.approx(2.0)

    def test_distance_based_envelope(
        self,
        make_distance: Callable[[np.ndarray, np.ndarray], Any],
        array_fn: Callable[..., Any],
    ) -> None:
        """Distance-based credal set agrees with the L1 clip envelope rule."""
        nominal = np.array([[0.5, 0.3, 0.2]])
        radius = np.array([0.1])
        cs = make_distance(nominal, radius)
        assert coverage(cs, array_fn([0])) == pytest.approx(1.0)
        assert coverage(cs, array_fn([2])) == pytest.approx(0.0)
        assert efficiency(cs) == pytest.approx(2.0)
        assert average_interval_width(cs) == pytest.approx(0.2)

    def test_probability_intervals(
        self,
        make_intervals: Callable[[np.ndarray, np.ndarray], Any],
        array_fn: Callable[..., Any],
    ) -> None:
        """Probability-intervals credal set covers/excludes by interval dominance."""
        lower = np.array([[0.1, 0.4, 0.05]])
        upper = np.array([[0.5, 0.6, 0.2]])
        cs = make_intervals(lower, upper)
        assert coverage(cs, array_fn([0])) == pytest.approx(1.0)
        assert coverage(cs, array_fn([2])) == pytest.approx(0.0)
        assert efficiency(cs) == pytest.approx(2.0)

    def test_probability_intervals_average_interval_width(
        self,
        make_intervals: Callable[[np.ndarray, np.ndarray], Any],
    ) -> None:
        """Mean width across both samples and classes."""
        lower = np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])
        upper = np.array([[0.4, 0.4, 0.4], [1.0, 1.0, 1.0]])
        cs = make_intervals(lower, upper)
        # widths: [[0.3, 0.2, 0.1], [1.0, 1.0, 1.0]]; mean = (0.6 + 3.0) / 6 = 0.6.
        assert average_interval_width(cs) == pytest.approx(0.6)
