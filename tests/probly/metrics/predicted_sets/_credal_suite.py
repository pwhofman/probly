"""Backend-agnostic credal-set suite for ``coverage`` and ``efficiency``.

The four registered representations are tested per backend through
:class:`CredalSuite`. Only credal-set families that exist on every supported
backend are parametrised here: ``ConvexCredalSet`` and
``ProbabilityIntervalsCredalSet``. Conformal-set tests live in
:mod:`tests.probly.metrics.predicted_sets._metrics_suite`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from probly.metrics import coverage, efficiency

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


class CredalSuite:
    """Tests that exercise the credal-set handlers per backend."""

    def test_convex_target_inside_hull(
        self,
        make_convex: Callable[[np.ndarray], Any],
        make_distribution: Callable[[np.ndarray], Any],
    ) -> None:
        """Target on the segment between two vertices is in the hull."""
        cs = make_convex(np.array([[[0.6, 0.3, 0.1], [0.4, 0.5, 0.1]]]))
        target = make_distribution(np.array([[0.5, 0.4, 0.1]]))  # midpoint
        assert coverage(cs, target) == pytest.approx(1.0)

    def test_convex_target_outside_hull(
        self,
        make_convex: Callable[[np.ndarray], Any],
        make_distribution: Callable[[np.ndarray], Any],
    ) -> None:
        """Target with a class probability the hull cannot reach is not covered."""
        cs = make_convex(np.array([[[0.6, 0.3, 0.1], [0.4, 0.5, 0.1]]]))
        # vertex max for class 2 is 0.1; target's 0.3 cannot be matched.
        target = make_distribution(np.array([[0.2, 0.5, 0.3]]))
        assert coverage(cs, target) == pytest.approx(0.0)

    def test_convex_efficiency_is_dominance_cardinality(
        self,
        make_convex: Callable[[np.ndarray], Any],
    ) -> None:
        """Efficiency = mean cardinality of the interval-dominance prediction set."""
        cs = make_convex(np.array([[[0.6, 0.3, 0.1], [0.4, 0.5, 0.1]]]))
        # lower = [0.4, 0.3, 0.1]; upper = [0.6, 0.5, 0.1]; mask = [T, T, F].
        assert efficiency(cs) == pytest.approx(2.0)

    def test_probability_intervals_target_inside(
        self,
        make_intervals: Callable[[np.ndarray, np.ndarray], Any],
        make_distribution: Callable[[np.ndarray], Any],
    ) -> None:
        """All classes satisfy ``lower[k] <= target[k] <= upper[k]``."""
        cs = make_intervals(np.array([[0.1, 0.1, 0.1]]), np.array([[0.3, 0.6, 0.4]]))
        target = make_distribution(np.array([[0.2, 0.5, 0.3]]))
        assert coverage(cs, target) == pytest.approx(1.0)

    def test_probability_intervals_target_outside_one_class(
        self,
        make_intervals: Callable[[np.ndarray, np.ndarray], Any],
        make_distribution: Callable[[np.ndarray], Any],
    ) -> None:
        """One class outside its interval is enough to fail coverage."""
        # Class 2: upper bound 0.25, target 0.3 -> 0.3 > 0.25, fails.
        cs = make_intervals(np.array([[0.1, 0.1, 0.1]]), np.array([[0.3, 0.6, 0.25]]))
        target = make_distribution(np.array([[0.2, 0.5, 0.3]]))
        assert coverage(cs, target) == pytest.approx(0.0)

    def test_probability_intervals_efficiency(
        self,
        make_intervals: Callable[[np.ndarray, np.ndarray], Any],
    ) -> None:
        """Efficiency = mean cardinality of the interval-dominance prediction set."""
        # max(lower) = 0.4; mask = upper >= 0.4 = [T, T, F]; cardinality = 2.
        cs = make_intervals(np.array([[0.1, 0.4, 0.05]]), np.array([[0.5, 0.6, 0.2]]))
        assert efficiency(cs) == pytest.approx(2.0)

    def test_probability_intervals_partial_coverage_across_instances(
        self,
        make_intervals: Callable[[np.ndarray, np.ndarray], Any],
        make_distribution: Callable[[np.ndarray], Any],
    ) -> None:
        """Two-instance fixture: one in, one out -> mean coverage 0.5."""
        cs = make_intervals(
            np.array([[0.1, 0.1, 0.1], [0.0, 0.0, 0.0]]),
            np.array([[0.5, 0.6, 0.4], [0.4, 0.4, 0.4]]),
        )
        target = make_distribution(np.array([[0.2, 0.5, 0.3], [0.5, 0.4, 0.1]]))
        assert coverage(cs, target) == pytest.approx(0.5)
