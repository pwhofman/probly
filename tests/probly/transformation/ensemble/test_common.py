"""Tests for commonpre ensemble generation."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from probly.predictor import Predictor
from probly.transformation import ensemble
from probly.transformation.ensemble.common import ensemble_generator, register


def test_unregistered_type_raises(dummy_predictor: Predictor) -> None:
    """No ensemble generator is registered for type, NotImplementedError must occur."""
    base = dummy_predictor
    with pytest.raises(
        NotImplementedError,
        match=f"No ensemble generator is registered for type {type(base)}",
    ):
        ensemble_generator(dummy_predictor)


def test_registered_generator_called(dummy_predictor: Predictor) -> None:
    """If the type is registered, the appropriate generator is called and its result is returned."""
    mock_generator = Mock()
    expected_result = object()
    mock_generator.return_value = expected_result

    register(type(dummy_predictor), mock_generator)

    result = ensemble(dummy_predictor, num_members=4)

    mock_generator.assert_called_once_with(
        dummy_predictor,
        num_members=4,
        reset_params=True,
    )
    assert result is expected_result
