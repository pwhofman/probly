"""Backend-agnostic tests for probly.method.laplace."""

from __future__ import annotations

import pytest

from probly.method.laplace import laplace


def test_laplace_invalid_pred_type_raises() -> None:
    """``laplace(model, pred_type='bogus')`` raises ValueError before dispatch."""
    with pytest.raises(ValueError, match="pred_type must be 'glm' or 'nn'"):
        laplace(object(), pred_type="bogus")
