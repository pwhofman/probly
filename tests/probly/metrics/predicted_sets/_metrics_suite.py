"""Backend-agnostic suite for the conformal-set evaluation metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from probly.evaluation import coverage, efficiency

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


class MetricsSuite:
    """Tests that run identically against every backend's conformal-set wrapper."""

    def test_onehot_perfect_coverage(self, make_onehot_set: Callable[[Any], Any], array_fn: Callable[..., Any]) -> None:
        """Coverage is 1.0 when every set contains the true class."""
        mask = array_fn([[True, False, False], [False, True, False], [False, False, True]])
        y_true = array_fn([0, 1, 2])
        assert coverage(make_onehot_set(mask), y_true) == pytest.approx(1.0)

    def test_onehot_zero_coverage(self, make_onehot_set: Callable[[Any], Any], array_fn: Callable[..., Any]) -> None:
        """Coverage is 0.0 when the true class is never selected."""
        mask = array_fn([[False, True, True], [True, False, True], [True, True, False]])
        y_true = array_fn([0, 1, 2])
        assert coverage(make_onehot_set(mask), y_true) == pytest.approx(0.0)

    def test_onehot_partial_coverage(self, make_onehot_set: Callable[[Any], Any], array_fn: Callable[..., Any]) -> None:
        """Coverage matches the manual fraction-of-correct-membership computation."""
        mask = array_fn([[True, False], [False, False], [True, True], [False, True]])
        y_true = array_fn([0, 0, 1, 1])
        assert coverage(make_onehot_set(mask), y_true) == pytest.approx(3 / 4)

    def test_onehot_efficiency(self, make_onehot_set: Callable[[Any], Any], array_fn: Callable[..., Any]) -> None:
        """Efficiency is the mean cardinality of the selected sets."""
        mask = array_fn([[True, False, False], [True, True, False], [True, True, True]])
        assert efficiency(make_onehot_set(mask)) == pytest.approx(2.0)

    def test_interval_perfect_coverage(
        self, make_interval_set: Callable[[Any], Any], array_fn: Callable[..., Any]
    ) -> None:
        """Coverage is 1.0 when every truth value is inside its interval."""
        intervals = array_fn([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=float)
        y_true = array_fn([0.5, 1.5, 2.5], dtype=float)
        assert coverage(make_interval_set(intervals), y_true) == pytest.approx(1.0)

    def test_interval_partial_coverage(
        self, make_interval_set: Callable[[Any], Any], array_fn: Callable[..., Any]
    ) -> None:
        """Coverage matches the manual inside-bounds fraction."""
        intervals = array_fn([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=float)
        y_true = array_fn([0.5, 5.0, 2.5], dtype=float)
        assert coverage(make_interval_set(intervals), y_true) == pytest.approx(2 / 3)

    def test_interval_efficiency(self, make_interval_set: Callable[[Any], Any], array_fn: Callable[..., Any]) -> None:
        """Efficiency is the mean width."""
        intervals = array_fn([[0.0, 1.0], [1.0, 4.0], [2.0, 4.0]], dtype=float)
        assert efficiency(make_interval_set(intervals)) == pytest.approx(2.0)

    def test_onehot_higher_rank_equivalence(
        self, make_onehot_set: Callable[[Any], Any], array_fn: Callable[..., Any]
    ) -> None:
        """Coverage on ``(B, N, C)`` matches the equivalent flat ``(B*N, C)`` call.

        Locks in the contract that any number of leading axes are treated as
        batch dimensions; guards against the regression where the legacy
        implementation hard-coded ``axis=1``.
        """
        rng = np.random.default_rng(0)
        flat_mask = rng.integers(0, 2, size=(6, 4)).astype(bool)
        flat_labels = rng.integers(0, 4, size=(6,))
        nested_mask = flat_mask.reshape(3, 2, 4)
        nested_labels = flat_labels.reshape(3, 2)

        flat = coverage(make_onehot_set(array_fn(flat_mask)), array_fn(flat_labels))
        nested = coverage(make_onehot_set(array_fn(nested_mask)), array_fn(nested_labels))
        assert nested == pytest.approx(flat)

    def test_onehot_efficiency_higher_rank(
        self, make_onehot_set: Callable[[Any], Any], array_fn: Callable[..., Any]
    ) -> None:
        """Efficiency on ``(B, N, C)`` matches the equivalent flat ``(B*N, C)`` call."""
        rng = np.random.default_rng(1)
        flat_mask = rng.integers(0, 2, size=(6, 5)).astype(bool)
        nested_mask = flat_mask.reshape(2, 3, 5)
        flat = efficiency(make_onehot_set(array_fn(flat_mask)))
        nested = efficiency(make_onehot_set(array_fn(nested_mask)))
        assert nested == pytest.approx(flat)

    def test_interval_higher_rank_equivalence(
        self, make_interval_set: Callable[[Any], Any], array_fn: Callable[..., Any]
    ) -> None:
        """Interval coverage on ``(B, N, 2)`` matches the equivalent flat ``(B*N, 2)`` call."""
        rng = np.random.default_rng(2)
        flat_intervals = np.sort(rng.random(size=(6, 2)), axis=-1)
        flat_y = rng.random(size=(6,))
        nested_intervals = flat_intervals.reshape(3, 2, 2)
        nested_y = flat_y.reshape(3, 2)

        flat = coverage(make_interval_set(array_fn(flat_intervals, dtype=float)), array_fn(flat_y, dtype=float))
        nested = coverage(make_interval_set(array_fn(nested_intervals, dtype=float)), array_fn(nested_y, dtype=float))
        assert nested == pytest.approx(flat)

    def test_interval_efficiency_higher_rank(
        self, make_interval_set: Callable[[Any], Any], array_fn: Callable[..., Any]
    ) -> None:
        """Interval efficiency on ``(B, N, 2)`` matches the equivalent flat ``(B*N, 2)`` call."""
        rng = np.random.default_rng(3)
        flat_intervals = np.sort(rng.random(size=(6, 2)), axis=-1)
        nested_intervals = flat_intervals.reshape(2, 3, 2)
        flat = efficiency(make_interval_set(array_fn(flat_intervals, dtype=float)))
        nested = efficiency(make_interval_set(array_fn(nested_intervals, dtype=float)))
        assert nested == pytest.approx(flat)
