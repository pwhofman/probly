"""test for torch bayesian models."""

from __future__ import annotations

import pytest
from torch import nn

from probly.transformation import bayesian

torch = pytest.importorskip("torch")


def test_bayesian_no_param_torch(torch_model_small_2d_2d: nn.Module) -> None:
    """Test that bayesian can be called with a torch model without prior or likelihood."""
    result = bayesian(torch_model_small_2d_2d)

    # Prüfen, dass ein Ergebnis zurückkommt
    assert result is not None
