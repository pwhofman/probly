"""Tests for commonpre ensemble generation."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from probly.method.ensemble import EnsemblePredictor, ensemble, ensemble_generator
from probly.predictor import Predictor


def test_unregistered_type_raises(dummy_predictor: Predictor) -> None:
    """No ensemble generator is registered for type, NotImplementedError must occur."""
    base = dummy_predictor
    with pytest.raises(
        NotImplementedError,
        match=f"No ensemble generator is registered for type {type(base)}",
    ):
        ensemble_generator(dummy_predictor, num_members=4)


def test_registered_generator_called(dummy_predictor: Predictor) -> None:
    """If the type is registered, the appropriate generator is called and its result is returned."""

    class GeneratedPredictor:
        pass

    mock_generator = Mock()
    expected_result = GeneratedPredictor()
    mock_generator.return_value = expected_result

    ensemble_generator.register(type(dummy_predictor), mock_generator)

    result = ensemble(dummy_predictor, num_members=4)

    mock_generator.assert_called_once_with(
        dummy_predictor,
        num_members=4,
        reset_params=True,
    )
    assert result is expected_result


def test_registered_generator_keeps_builtin_list_of_predictors(dummy_predictor: Predictor) -> None:
    """Autocast should not wrap proper list ensembles that already satisfy EnsemblePredictor."""
    mock_generator = Mock(return_value=[dummy_predictor, dummy_predictor])

    ensemble_generator.register(type(dummy_predictor), mock_generator)

    result = ensemble(dummy_predictor, num_members=2)

    assert type(result) is list
    assert isinstance(result, EnsemblePredictor)


def test_registered_generator_autocasts_builtin_list_of_non_predictors(dummy_predictor: Predictor) -> None:
    """Autocast should register future list-like ensemble outputs that are not already valid ensembles."""
    mock_generator = Mock(return_value=[object()])

    ensemble_generator.register(type(dummy_predictor), mock_generator)

    result = ensemble(dummy_predictor, num_members=1)

    assert type(result) is not list
    assert isinstance(result, list)
    assert isinstance(result, EnsemblePredictor)
