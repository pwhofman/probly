"""Tests for commonpre ensemble generation."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from probly.predictor import Predictor
from probly.transformation import ensemble
from probly.transformation.ensemble.common import register


def test_unregistered_type_raises(dummy_predictor: Predictor) -> None:
    """No ensemble generator is registered for type, NotImplementedError must occur."""
    with pytest.raises(TypeError):
        ensemble(dummy_predictor, n_members=2)


def test_registered_generator_called(dummy_predictor: Predictor) -> None:
    """If the type is registered, the appropriate generator is called and its result ist returned."""
    mock_generator = Mock()
    expected_result = object()
    mock_generator.return_value = expected_result

    register(type(dummy_predictor), mock_generator)

    result = ensemble(dummy_predictor, n_members=4)

    mock_generator.assert_called_once_with(dummy_predictor, n_members=4)
    assert result is expected_result
    """kk"""
