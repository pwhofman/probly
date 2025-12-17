"""Tests for probly.transformation.evidential.classification.torch."""

from __future__ import annotations

import importlib

import torch

from probly.transformation.evidential.classification import common


def test_append_activation_torch_adds_softplus_and_handles_input() -> None:
    """Check that the torch-specific evidential classification registration appends.

    a Softplus layer and that the wrapped module can process input.
    """
    # Importing the module to ensure the torch-specific registration is executed
    importlib.import_module(
        "probly.transformation.evidential.classification.torch",
    )

    base = torch.nn.Linear(4, 2)
    wrapped = common.evidential_classification(base)

    # Check structure of the wrapped module
    assert isinstance(wrapped, torch.nn.Sequential)
    assert len(wrapped) == 2
    assert isinstance(wrapped[1], torch.nn.Softplus)

    # Check that the wrapped module can process input correctly
    x = torch.ones(3, 4)  # batch of 3, input dim 4
    y = wrapped(x)

    # Output should have the correct shape
    assert y.shape == (3, 2)

    # Because of Softplus, outputs should be non-negative
    assert torch.all(y >= 0)


def test_append_activation_torch_works_with_sequential_base() -> None:
    """Check that the registration also works when the base model is already a Sequential module."""
    # Again import to ensure registration
    importlib.import_module(
        "probly.transformation.evidential.classification.torch",
    )

    base = torch.nn.Sequential(
        torch.nn.Linear(4, 2),
    )
    wrapped = common.evidential_classification(base)

    # Last layer should be a Softplus that was appended
    assert isinstance(wrapped, torch.nn.Sequential)
    assert isinstance(wrapped[-1], torch.nn.Softplus)

    # Check that the wrapped module can process input
    x = torch.randn(5, 4)
    y = wrapped(x)

    assert y.shape == (5, 2)
