"""Tests for single-prediction conversion semantics."""

from __future__ import annotations

import numpy as np
import pytest

from probly.method.ensemble import EnsemblePredictor  # noqa: F401
from probly.predictor import predict_single, to_single_prediction
from probly.representation import CanonicalRepresentation
from probly.representation.credal_set.array import (
    ArrayDistanceBasedCredalSet,
    ArrayProbabilityIntervalsCredalSet,
)
from probly.representation.distribution import ArrayCategoricalDistribution
from probly.representation.distribution.array_dirichlet import ArrayDirichletDistribution
from probly.representation.sample import ArraySample


class _ListPredictor:
    def predict(self, x: int) -> list[int]:
        return [x, x + 1]


def test_predict_single_defaults_to_predict_for_plain_outputs() -> None:
    assert predict_single(_ListPredictor(), 1) == [1, 2]


def test_to_single_prediction_reduces_sample_to_mean() -> None:
    sample = ArraySample.from_iterable([np.array([1.0, 3.0]), np.array([3.0, 5.0])], sample_axis=0)

    assert isinstance(sample, CanonicalRepresentation)
    np.testing.assert_allclose(sample.canonical_element, np.array([2.0, 4.0]))
    np.testing.assert_allclose(to_single_prediction(sample), np.array([2.0, 4.0]))


def test_predict_single_reduces_ensemble_predictions_to_sample_mean() -> None:
    predictors = [lambda x: x, lambda x: x + 2.0]

    np.testing.assert_allclose(predict_single(predictors, np.array([1.0, 2.0])), np.array([2.0, 3.0]))


def test_to_single_prediction_reduces_array_dirichlet_to_expected_categorical_distribution() -> None:
    distribution = ArrayDirichletDistribution(np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 4.0]]))

    single = to_single_prediction(distribution)

    assert isinstance(single, ArrayCategoricalDistribution)
    np.testing.assert_allclose(single.probabilities, np.array([[1 / 6, 2 / 6, 3 / 6], [2 / 8, 2 / 8, 4 / 8]]))


def test_to_single_prediction_reduces_probability_intervals_to_center_distribution() -> None:
    credal_set = ArrayProbabilityIntervalsCredalSet(
        lower_bounds=np.array([[0.1, 0.2, 0.3]]),
        upper_bounds=np.array([[0.3, 0.4, 0.5]]),
    )

    single = to_single_prediction(credal_set)

    assert isinstance(single, ArrayCategoricalDistribution)
    np.testing.assert_allclose(single.unnormalized_probabilities, np.array([[7 / 30, 1 / 3, 13 / 30]]))


def test_probability_interval_center_is_a_valid_distribution_for_vacuous_intervals() -> None:
    credal_set = ArrayProbabilityIntervalsCredalSet(
        lower_bounds=np.array([[0.0, 0.0, 0.0]]),
        upper_bounds=np.array([[1.0, 1.0, 1.0]]),
    )

    single = to_single_prediction(credal_set)

    assert isinstance(single, ArrayCategoricalDistribution)
    np.testing.assert_allclose(single.unnormalized_probabilities, np.array([[1 / 3, 1 / 3, 1 / 3]]))


def test_to_single_prediction_reduces_distance_based_credal_set_to_nominal_distribution() -> None:
    nominal = ArrayCategoricalDistribution(np.array([[0.2, 0.3, 0.5]]))
    credal_set = ArrayDistanceBasedCredalSet(nominal=nominal, radius=np.array([0.1]))

    assert to_single_prediction(credal_set) is nominal


def test_predict_single_reduces_torch_dirichlet_representation_to_expected_categorical_distribution() -> None:
    torch = pytest.importorskip("torch")
    torch_categorical = pytest.importorskip("probly.representation.distribution.torch_categorical")
    torch_dirichlet = pytest.importorskip("probly.representation.distribution.torch_dirichlet")

    class TorchDirichletPredictor:
        def predict(self, alphas: torch.Tensor) -> object:
            return torch_dirichlet.TorchDirichletDistribution(alphas)

    single = predict_single(TorchDirichletPredictor(), torch.tensor([[1.0, 2.0, 3.0]]))

    assert isinstance(single, torch_categorical.TorchCategoricalDistribution)
    assert torch.allclose(single.probabilities, torch.tensor([[1 / 6, 2 / 6, 3 / 6]]))


def test_to_single_prediction_uses_canonical_representation_protocol() -> None:
    class CustomPrediction:
        def __init__(self, value: int) -> None:
            self.value = value

        @property
        def canonical_element(self) -> int:
            return self.value

    assert isinstance(CustomPrediction(3), CanonicalRepresentation)
    assert to_single_prediction(CustomPrediction(3)) == 3


def test_predict_single_uses_custom_predictor_registration() -> None:
    class CustomPredictor:
        def predict(self) -> str:
            return "predict"

    @predict_single.register(CustomPredictor)
    def _(predictor: CustomPredictor) -> str:
        del predictor
        return "predict_single"

    assert predict_single(CustomPredictor()) == "predict_single"
