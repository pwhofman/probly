"""Tests for the common evidential classification transformation."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from probly.predictor import Predictor
from probly.transformation.evidential import evidential_classification
from probly.transformation.evidential.classification.common import register


def test_invalid_base(dummy_predictor: Predictor) -> None:
    """Test that an error is raised for an invalid base predictor.

    This function validates that a NotImplementedError is raised when
    an invalid base is used.

    Args:
       dummy_predictor: returns a dummy predictor

    Raises:
       NotImplementedError: If the base predictor type is not registered.
    """
    base = dummy_predictor
    with pytest.raises(
        NotImplementedError,
        match=f"No evidential classification appender registered for type {type(base)}",
    ):
        evidential_classification(base)  # type: ignore[type-var]


def test_valid_base(dummy_predictor: Predictor) -> None:
    """Test that no error is raised for a valid base predictor.

    This function ensures that no error is raised when a valid base
    predictor is used.

    Args:
       dummy_predictor: returns a dummy predictor
    """
    mock_generator = Mock()
    expected_result = object()
    mock_generator.return_value = expected_result

    register(type(dummy_predictor), mock_generator)

    result = evidential_classification(dummy_predictor)

    mock_generator.assert_called_once_with(dummy_predictor)
    assert result is expected_result
