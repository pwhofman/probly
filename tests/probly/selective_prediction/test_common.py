"""Backend-agnostic tests for selective predictors, run with NumPy."""

from __future__ import annotations

from typing import Any, override

import numpy as np
import pytest

from probly.decider import categorical_from_mean
from probly.predictor import predict
from probly.quantification import EpistemicUncertainty, TotalUncertainty, quantify
from probly.representation.distribution import create_categorical_distribution
from probly.representation.distribution.numpy_categorical import (
    NumpyCategoricalDistributionSample,
    NumpyProbabilityCategoricalDistribution,
)
from probly.representation.distribution.numpy_gaussian import (
    NumpyGaussianDistribution,
    NumpyGaussianDistributionSample,
)
from probly.representer import Representer, representer
from probly.selective_prediction import (
    SelectivePrediction,
    SelectivePredictor,
    Selector,
    ThresholdSelector,
)

PROBABILITIES = np.array([[0.9, 0.1], [0.5, 0.5], [0.6, 0.4], [0.99, 0.01]])
ENTROPIES = -(PROBABILITIES * np.log(PROBABILITIES)).sum(axis=-1)


class _RepresentationModel:
    """Uncertainty-aware stub model that returns a fixed representation."""

    def __init__(self, representation: Any) -> None:  # noqa: ANN401
        self.representation = representation

    def predict_representation(self, _x: object) -> Any:  # noqa: ANN401
        return self.representation


def _model() -> _RepresentationModel:
    return _RepresentationModel(create_categorical_distribution(PROBABILITIES))


def _ensemble_sample() -> NumpyCategoricalDistributionSample:
    # Two members and two instances; the members agree on the second instance only.
    probabilities = np.array(
        [
            [[0.9, 0.1], [0.7, 0.3]],
            [[0.2, 0.8], [0.7, 0.3]],
        ]
    )
    return NumpyCategoricalDistributionSample(
        array=NumpyProbabilityCategoricalDistribution(probabilities),
        sample_axis=0,
    )


def _regression_sample() -> NumpyGaussianDistributionSample:
    means = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 2.5]])
    variances = np.array([[1.0, 0.5, 0.1], [2.0, 0.5, 0.2]])
    return NumpyGaussianDistributionSample(NumpyGaussianDistribution(mean=means, var=variances), sample_axis=0)


def test_default_criterion_is_total_uncertainty_of_quantify() -> None:
    model = _model()
    result = SelectivePredictor(model, ThresholdSelector(0.5)).predict(None)
    representation = model.predict_representation(None)

    assert isinstance(result, SelectivePrediction)
    np.testing.assert_allclose(result.uncertainty, ENTROPIES)
    np.testing.assert_allclose(result.uncertainty, quantify(representation).total)
    np.testing.assert_allclose(
        result.decision.probabilities,
        categorical_from_mean(representation).probabilities,
    )


def test_accepted_is_uncertainty_at_most_threshold() -> None:
    result = SelectivePredictor(_model(), ThresholdSelector(0.5)).predict(None)

    np.testing.assert_array_equal(result.accepted, result.uncertainty <= 0.5)
    np.testing.assert_array_equal(result.accepted, [True, False, False, True])
    assert result.coverage == 0.5


def test_tie_at_threshold_is_accepted() -> None:
    result = SelectivePredictor(_model(), ThresholdSelector(float(ENTROPIES[2]))).predict(None)

    assert result.accepted[2]


@pytest.mark.parametrize(("threshold", "coverage"), [(np.inf, 1.0), (-np.inf, 0.0)])
def test_infinite_thresholds_accept_all_or_none(threshold: float, coverage: float) -> None:
    result = SelectivePredictor(_model(), ThresholdSelector(threshold)).predict(None)

    assert result.coverage == coverage


def test_nan_threshold_raises() -> None:
    with pytest.raises(ValueError, match="NaN"):
        ThresholdSelector(float("nan"))


def test_threshold_selector_works_on_plain_arrays() -> None:
    selector = ThresholdSelector(0.5)
    uncertainty = np.array([0.1, 0.5, 0.7, np.nan])

    np.testing.assert_array_equal(selector.select(uncertainty), [True, True, False, False])
    np.testing.assert_array_equal(selector(uncertainty), selector.select(uncertainty))


def test_predictor_uses_selector_on_quantified_uncertainty() -> None:
    selector = ThresholdSelector(0.5)
    result = SelectivePredictor(_model(), selector).predict(None)

    np.testing.assert_array_equal(result.accepted, selector.select(ENTROPIES))


@pytest.mark.parametrize("notion", ["total", "aleatoric", "epistemic"])
def test_notion_selects_component_of_decomposition(notion: str) -> None:
    sample = _ensemble_sample()
    predictor = SelectivePredictor(_RepresentationModel(sample), ThresholdSelector(0.2), notion=notion)

    result = predictor.predict(None)

    np.testing.assert_allclose(result.uncertainty, quantify(sample)[notion])


def test_epistemic_notion_separates_disagreement_from_noise() -> None:
    result = SelectivePredictor(
        _RepresentationModel(_ensemble_sample()), ThresholdSelector(0.1), notion="epistemic"
    ).predict(None)

    assert result.uncertainty[0] > 0.1
    np.testing.assert_allclose(result.uncertainty[1], 0.0, atol=1e-12)
    np.testing.assert_array_equal(result.accepted, [False, True])


@pytest.mark.parametrize("notion", ["EU", "eu", EpistemicUncertainty])
def test_notion_accepts_aliases_and_classes(notion: Any) -> None:  # noqa: ANN401
    predictor = SelectivePredictor(_RepresentationModel(_ensemble_sample()), ThresholdSelector(0.2), notion=notion)

    assert predictor.notion is EpistemicUncertainty


def test_default_notion_is_total() -> None:
    assert SelectivePredictor(_model(), ThresholdSelector(0.2)).notion is TotalUncertainty


def test_invalid_notion_raises() -> None:
    with pytest.raises(ValueError, match="notion"):
        SelectivePredictor(_model(), ThresholdSelector(0.2), notion="bogus")


def test_notion_missing_from_decomposition_raises() -> None:
    predictor = SelectivePredictor(_model(), ThresholdSelector(0.2), notion="epistemic")

    with pytest.raises(KeyError, match="EpistemicUncertainty"):
        predictor.predict(None)


@pytest.mark.parametrize("notion", ["total", "aleatoric", "epistemic"])
def test_regression_criterion_comes_from_quantify(notion: str) -> None:
    sample = _regression_sample()
    result = SelectivePredictor(
        _RepresentationModel(sample), ThresholdSelector(1.0), notion=notion, decider=lambda rep: rep
    ).predict(None)

    np.testing.assert_allclose(result.uncertainty, quantify(sample)[notion])


def test_custom_decider_is_used() -> None:
    result = SelectivePredictor(
        _model(), ThresholdSelector(0.2), decider=lambda rep: rep.probabilities.argmax(-1)
    ).predict(None)

    np.testing.assert_array_equal(result.decision, [0, 0, 0, 0])


def test_predict_dispatch_and_call_agree_with_method() -> None:
    sp = SelectivePredictor(_model(), ThresholdSelector(0.5))

    via_method = sp.predict(None)
    via_dispatch = predict(sp, None)
    via_call = sp(None)

    for other in (via_dispatch, via_call):
        np.testing.assert_array_equal(other.accepted, via_method.accepted)
        np.testing.assert_allclose(other.uncertainty, via_method.uncertainty)


class _SamplingModel:
    """Stub model whose representer requires a number of samples, like a sampling-based model."""


class _SamplingRepresenter(Representer):
    """Representer for the stub sampling model that records its number of samples."""

    def __init__(self, predictor: _SamplingModel, num_samples: int) -> None:
        super().__init__(predictor)
        self.num_samples = num_samples

    @override
    def represent(self, _x: object) -> Any:
        return create_categorical_distribution(PROBABILITIES)


representer.register(_SamplingModel, _SamplingRepresenter)


def test_representer_kwargs_are_passed_to_the_representer() -> None:
    sp = SelectivePredictor(_SamplingModel(), ThresholdSelector(0.5), representer_kwargs={"num_samples": 3})

    assert isinstance(sp.representer, _SamplingRepresenter)
    assert sp.representer.num_samples == 3
    np.testing.assert_allclose(sp.predict(None).uncertainty, ENTROPIES)


def test_missing_representer_kwargs_raise() -> None:
    with pytest.raises(TypeError, match="num_samples"):
        SelectivePredictor(_SamplingModel(), ThresholdSelector(0.5))


def test_selector_base_class_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        Selector()


class _LeastUncertainSelector(Selector):
    """Selector that overrides only `select` and keeps the k least uncertain instances."""

    def __init__(self, k: int) -> None:
        self.k = k

    @override
    def select(self, uncertainty: Any) -> np.ndarray:
        accepted = np.zeros(uncertainty.shape, dtype=bool)
        accepted[np.argsort(uncertainty, kind="stable")[: self.k]] = True
        return accepted


def test_custom_selector_runs_through_predictor_pipeline() -> None:
    result = SelectivePredictor(_model(), _LeastUncertainSelector(k=1)).predict(None)

    assert isinstance(result, SelectivePrediction)
    np.testing.assert_array_equal(result.accepted, [False, False, False, True])
    assert result.coverage == 0.25
