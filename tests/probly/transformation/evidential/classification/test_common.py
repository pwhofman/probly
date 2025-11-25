"""probly.transformation.evidential.classification.common."""

from __future__ import annotations

import importlib

import pytest
import torch

# Import the common module once for all tests

common = importlib.import_module(
    "probly.transformation.evidential.classification.common",
)


def test_evidential_classification_handles_valid_torch_module() -> None:
    """Check that evidential_classification wraps a valid torch module.

    and that the wrapped module can process input data.
    """
    base = torch.nn.Linear(4, 2)
    wrapped = common.evidential_classification(base)

    # The wrapped module should be callable and accept tensors of the correct shape
    x = torch.randn(3, 4)
    y = wrapped(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape == (3, 2)


def test_evidential_classification_rejects_invalid_input() -> None:
    """Check that evidential_classification fails on invalid input types."""
    invalid_input = "not_a_module"

    with pytest.raises((TypeError, ValueError)):
        common.evidential_classification(invalid_input)  # type: ignore[arg-type]


def test_register_rejects_invalid_arguments() -> None:
    """Check that register fails when called with invalid arguments."""
    # We assume that passing obviously wrong types should raise an error.
    # Exact signature is defined in the implementation, but incorrect types
    # must not be silently accepted.

    with pytest.raises(TypeError):
        common.register(123)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        common.register(None)  # type: ignore[arg-type]
