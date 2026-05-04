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

    def test_convex_average_interval_width(self, make_convex: Callable[[np.ndarray], Any]) -> None:
        """Mean per-class width of the vertex-min/vertex-max envelope of a convex hull.

        With vertices ``[[0.6, 0.3, 0.1], [0.4, 0.5, 0.1]]`` the envelope is
        ``lower=[0.4, 0.3, 0.1]``, ``upper=[0.6, 0.5, 0.1]``, giving widths
        ``[0.2, 0.2, 0.0]`` and mean ``0.4 / 3``.
        """
        probs = np.array([[[0.6, 0.3, 0.1], [0.4, 0.5, 0.1]]])
        cs = make_convex(probs)
        assert average_interval_width(cs) == pytest.approx(0.4 / 3)

    def test_intervals_higher_rank(
        self,
        make_intervals: Callable[[np.ndarray, np.ndarray], Any],
        array_fn: Callable[..., Any],
    ) -> None:
        """Probability-intervals coverage on ``(B, N, C)`` matches the flat ``(B*N, C)`` call."""
        rng = np.random.default_rng(11)
        flat_lower = rng.random(size=(6, 4)) * 0.3
        flat_upper = flat_lower + rng.random(size=(6, 4)) * 0.3
        flat_y = rng.integers(0, 4, size=(6,))
        nested_lower = flat_lower.reshape(3, 2, 4)
        nested_upper = flat_upper.reshape(3, 2, 4)
        nested_y = flat_y.reshape(3, 2)

        flat = coverage(make_intervals(flat_lower, flat_upper), array_fn(flat_y))
        nested = coverage(make_intervals(nested_lower, nested_upper), array_fn(nested_y))
        assert nested == pytest.approx(flat)
        assert efficiency(make_intervals(flat_lower, flat_upper)) == pytest.approx(
            efficiency(make_intervals(nested_lower, nested_upper))
        )

    @pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide:RuntimeWarning")
    def test_empty_batch_returns_nan(
        self,
        make_intervals: Callable[[np.ndarray, np.ndarray], Any],
        array_fn: Callable[..., Any],
    ) -> None:
        """Coverage and efficiency on an empty batch return ``nan`` consistently per backend."""
        cs = make_intervals(np.zeros((0, 3)), np.ones((0, 3)))
        cov = coverage(cs, array_fn(np.zeros((0,), dtype=int)))
        eff = efficiency(cs)
        assert np.isnan(cov)
        assert np.isnan(eff)
