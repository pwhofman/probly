"""Tests for the numpy/common implementation of ``calculate_quantile``."""

from __future__ import annotations

import numpy as np
import pytest


class TestQuantileNumpy:
    """`calculate_quantile` and weighted variant on numpy arrays."""

    def test_basic_quantile(self) -> None:
        from probly.utils.quantile import calculate_quantile  # noqa: PLC0415

        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        q = calculate_quantile(scores, alpha=0.1)
        # alpha=0.1 -> q_level ≈ 1.08 -> capped at 1 -> highest score.
        assert q == pytest.approx(0.5)

    def test_alpha_out_of_range_raises(self) -> None:
        from probly.utils.quantile import calculate_quantile  # noqa: PLC0415

        with pytest.raises(ValueError, match="alpha must be in"):
            calculate_quantile(np.array([0.1, 0.2]), alpha=1.5)
        with pytest.raises(ValueError, match="alpha must be in"):
            calculate_quantile(np.array([0.1, 0.2]), alpha=-0.5)

    def test_empty_scores_raises(self) -> None:
        from probly.utils.quantile import calculate_quantile  # noqa: PLC0415

        with pytest.raises(ValueError, match="empty"):
            calculate_quantile(np.array([]), alpha=0.1)

    def test_weighted_quantile_unweighted_matches_unweighted(self) -> None:
        from probly.utils.quantile._common import calculate_weighted_quantile  # noqa: PLC0415

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = calculate_weighted_quantile(values, 0.5)
        assert result == pytest.approx(3.0)

    def test_weighted_quantile_with_weights(self) -> None:
        from probly.utils.quantile._common import calculate_weighted_quantile  # noqa: PLC0415

        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 0.0, 0.0])  # all weight on 1.0
        result = calculate_weighted_quantile(values, 0.5, sample_weight=weights)
        assert result == pytest.approx(1.0)
