"""Tests for the total-variation nonconformity score on torch.

Covers torch dispatch parity with the numpy implementation and the
requirement that the numpy callable form rejects calls without
``y_true``.
"""

from __future__ import annotations

import numpy as np
import pytest


def _torch():
    """Return torch module or skip the calling test."""
    return pytest.importorskip("torch")


class TestTotalVariationTorch:
    """Total-variation torch dispatch."""

    def test_matches_numpy(self) -> None:
        torch = _torch()
        from probly.conformal_scores import tv_score_func  # noqa: PLC0415

        y_pred = np.array([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])
        y_true = np.array([0, 2])
        expected = tv_score_func(y_pred, y_true)
        result = tv_score_func(torch.tensor(y_pred), torch.tensor(y_true))
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)


class TestTotalVariationCallable:
    def test_callable_requires_y_true(self) -> None:
        from probly.conformal_scores.total_variation._common import tv_score  # noqa: PLC0415

        with pytest.raises(ValueError, match="y_true is required"):
            tv_score(np.array([[0.2, 0.8]]))
