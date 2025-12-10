"""Test for ensemble models."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from probly.predictor import Predictor
from probly.transformation import ensemble
from probly.transformation.ensemble.common import ensemble_generator, register


def test_invalid_base_module(dummy_predictor: Predictor) -> None:
    """Base module without generator should throw an NotImplementedError."""
    base = dummy_predictor
    with pytest.raises(NotImplementedError):
        ensemble_generator(base)


def test_not_implemented(dummy_predictor: Predictor) -> None:
    """Generate ensemble for unregistered type leads to error."""
    predictor_type = type(dummy_predictor)
    expected_message = f"No ensemble generator is registered for type {predictor_type}"

    with pytest.raises(NotImplementedError) as exception:
        ensemble_generator(dummy_predictor)

    assert expected_message in str(exception.value)


def test_registered_generator_is_used(dummy_predictor: Predictor) -> None:
    # Arrange test variables.
    generator_mock = Mock()
    output = object()
    generator_mock.return_value = output

    register(type(dummy_predictor), generator_mock)

    result = ensemble(dummy_predictor, num_members=3)

    generator_mock.assert_called_once()
    call_args, call_kwargs = generator_mock.call_args

    assert call_args[0] is dummy_predictor
    assert call_kwargs["num_members"] == 3
    assert call_kwargs["reset_params"] is True
    assert result is output
