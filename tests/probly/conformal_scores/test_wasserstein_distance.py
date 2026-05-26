"""Tests for the Wasserstein-distance nonconformity score on torch.

Covers torch dispatch parity with the numpy implementation, the zero
self-distance property, and the requirement that the callable form
rejects calls without ``y_true``.
"""

from __future__ import annotations

import numpy as np
import pytest


def _torch():
    """Return torch module or skip the calling test."""
    return pytest.importorskip("torch")


class TestWassersteinDistanceTorch:
    """Wasserstein-distance torch dispatch."""

    def test_matches_numpy(self) -> None:
        torch = _torch()
        from probly.conformal_scores.wasserstein_distance._common import (  # noqa: PLC0415
            wasserstein_distance_score,
            wasserstein_distance_score_func,
        )

        y_pred = np.array([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])
        y_true = np.array([0, 2])
        expected = wasserstein_distance_score_func(y_pred, y_true)
        result = wasserstein_distance_score_func(torch.tensor(y_pred), torch.tensor(y_true))
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)
        with pytest.raises(ValueError, match="y_true is required"):
            wasserstein_distance_score(torch.tensor(y_pred))

    def test_self_distance_is_zero(self) -> None:
        torch = _torch()
        from probly.conformal_scores.wasserstein_distance._common import (  # noqa: PLC0415
            wasserstein_distance_score_func,
        )

        p = torch.tensor([[0.3, 0.4, 0.3]])
        result = wasserstein_distance_score_func(p, p)
        assert result.item() == pytest.approx(0.0, abs=1e-6)
