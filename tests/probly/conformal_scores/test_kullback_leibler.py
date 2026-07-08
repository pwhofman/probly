"""Tests for the Kullback-Leibler nonconformity score on torch.

Covers torch dispatch parity with the numpy implementation, the zero
self-divergence property, and the requirement that the callable form
rejects calls without ``y_true``.
"""

from __future__ import annotations

import numpy as np
import pytest


def _torch():
    """Return torch module or skip the calling test."""
    return pytest.importorskip("torch")


class TestKLDivergenceTorch:
    """Kullback-Leibler torch dispatch."""

    def test_matches_numpy(self) -> None:
        torch = _torch()
        from probly.conformal_scores import kl_divergence_score, kl_divergence_score_func  # noqa: PLC0415

        y_pred = np.array([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])
        y_true = np.array([0, 2])
        expected = kl_divergence_score_func(y_pred, y_true)
        result = kl_divergence_score_func(torch.tensor(y_pred), torch.tensor(y_true))
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)
        with pytest.raises(ValueError, match="y_true is required"):
            kl_divergence_score(torch.tensor(y_pred))

    def test_self_divergence_is_zero(self) -> None:
        torch = _torch()
        from probly.conformal_scores import kl_divergence_score_func  # noqa: PLC0415

        p = torch.tensor([[0.3, 0.4, 0.3]])
        result = kl_divergence_score_func(p, p)
        assert result.item() == pytest.approx(0.0, abs=1e-6)
