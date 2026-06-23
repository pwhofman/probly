"""Tests for the inner-product nonconformity score on torch.

Covers torch dispatch parity with the numpy implementation for both
class-index and one-hot label encodings, and the requirement that the
callable form rejects calls without ``y_true``.
"""

from __future__ import annotations

import numpy as np
import pytest


def _torch():
    """Return torch module or skip the calling test."""
    return pytest.importorskip("torch")


class TestInnerProductTorch:
    """Inner-product score torch dispatch."""

    def test_matches_numpy_with_class_indices(self) -> None:
        torch = _torch()
        from probly.conformal_scores import inner_product_score, inner_product_score_func  # noqa: PLC0415

        y_pred = np.array([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])
        y_true = np.array([1, 2])
        expected = inner_product_score_func(y_pred, y_true)
        result = inner_product_score_func(torch.tensor(y_pred), torch.tensor(y_true))
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)
        # callable form requires y_true
        with pytest.raises(ValueError, match="y_true is required"):
            inner_product_score(torch.tensor(y_pred))

    def test_matches_numpy_with_one_hot(self) -> None:
        torch = _torch()
        from probly.conformal_scores import inner_product_score_func  # noqa: PLC0415

        y_pred = torch.tensor([[0.2, 0.8]])
        y_true = torch.tensor([[1.0, 0.0]])
        result = inner_product_score_func(y_pred, y_true)
        # 1 - 0.2*1.0 - 0.8*0.0 = 0.8
        assert torch.allclose(result, torch.tensor([0.8]), atol=1e-6)
