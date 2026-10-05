"""Backend-agnostic tests for selective predictors, run with NumPy."""

from __future__ import annotations

from typing import Any, override

import numpy as np
import pytest

from probly.calibrator import Calibrator, calibrate
from probly.decider import categorical_from_mean
from probly.predictor import predict
from probly.quantification import (
    BrierLoss,
    EpistemicUncertainty,
    LogLoss,
    SecondOrderScoringRuleDecomposition,
    SecondOrderZeroOneDecomposition,
    TotalUncertainty,
    ZeroOneLoss,
    quantify,
)
from probly.representation.distribution import (
    create_categorical_distribution,
    create_dirichlet_distribution_from_alphas,
)
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
    CoverageSelector,
    SelectivePrediction,
    SelectivePredictor,
    Selector,
    ThresholdSelector,
)

PROBABILITIES = np.array([[0.9, 0.1], [0.5, 0.5], [0.6, 0.4], [0.99, 0.01]])
MAX_PROB_COMPLEMENTS = 1.0 - PROBABILITIES.max(axis=-1)
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


def test_default_criterion_is_one_minus_max_probability() -> None:
    model = _model()
    result = SelectivePredictor(model, ThresholdSelector(0.5)).predict(None)
    representation = model.predict_representation(None)

    assert isinstance(result, SelectivePrediction)
    np.testing.assert_allclose(result.uncertainty, MAX_PROB_COMPLEMENTS)
    np.testing.assert_allclose(
        result.decision.probabilities,
        categorical_from_mean(representation).probabilities,
    )


def test_accepted_is_uncertainty_at_most_threshold() -> None:
    result = SelectivePredictor(_model(), ThresholdSelector(0.3)).predict(None)

    np.testing.assert_array_equal(result.accepted, result.uncertainty <= 0.3)
    np.testing.assert_array_equal(result.accepted, [True, False, False, True])
    assert result.coverage == 0.5


def test_tie_at_threshold_is_accepted() -> None:
    result = SelectivePredictor(_model(), ThresholdSelector(float(MAX_PROB_COMPLEMENTS[2]))).predict(None)

    assert result.accepted[2]


@pytest.mark.parametrize(("threshold", "coverage"), [(np.inf, 1.0), (-np.inf, 0.0)])
def test_infinite_thresholds_accept_all_or_none(threshold: float, coverage: float) -> None:
    result = SelectivePredictor(_model(), ThresholdSelector(threshold)).predict(None)

    assert result.coverage == coverage


def test_coverage_of_empty_batch_is_nan() -> None:
    result = SelectivePrediction(decision=None, uncertainty=np.zeros(0), accepted=np.zeros(0, dtype=bool))

    assert np.isnan(result.coverage)


def test_nan_threshold_raises() -> None:
    with pytest.raises(ValueError, match="NaN"):
        ThresholdSelector(float("nan"))


def test_threshold_selector_works_on_plain_arrays() -> None:
    selector = ThresholdSelector(0.5)
    uncertainty = np.array([0.1, 0.5, 0.7, np.nan])

    np.testing.assert_array_equal(selector.select(uncertainty), [True, True, False, False])
    np.testing.assert_array_equal(selector(uncertainty), selector.select(uncertainty))


@pytest.mark.parametrize("uncertainty", [np.float64(0.1), np.zeros((2, 2))])
def test_threshold_selector_rejects_non_one_dimensional_uncertainty(uncertainty: np.ndarray) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        ThresholdSelector(0.5).select(uncertainty)


def test_threshold_selector_rejects_non_array_uncertainty() -> None:
    with pytest.raises(TypeError, match="array"):
        ThresholdSelector(0.5).select([0.1, 0.9])


def test_predictor_rejects_criterion_with_extra_axes() -> None:
    predictor = SelectivePredictor(_model(), ThresholdSelector(0.5), notion=lambda rep: rep.probabilities)

    with pytest.raises(ValueError, match=r"shape \(4, 2\)"):
        predictor.predict(None)


def test_predictor_uses_selector_on_criterion() -> None:
    selector = ThresholdSelector(0.3)
    result = SelectivePredictor(_model(), selector).predict(None)

    np.testing.assert_array_equal(result.accepted, selector.select(MAX_PROB_COMPLEMENTS))


@pytest.mark.parametrize("notion", ["total", "aleatoric", "epistemic"])
def test_notion_selects_component_of_decomposition(notion: str) -> None:
    sample = _ensemble_sample()
    predictor = SelectivePredictor(_RepresentationModel(sample), ThresholdSelector(0.2), notion=notion)

    result = predictor.predict(None)

    np.testing.assert_allclose(result.uncertainty, SecondOrderZeroOneDecomposition(sample)[notion])


def test_epistemic_notion_separates_disagreement_from_noise() -> None:
    result = SelectivePredictor(
        _RepresentationModel(_ensemble_sample()), ThresholdSelector(0.1), notion="epistemic"
    ).predict(None)

    assert result.uncertainty[0] > 0.1
    assert result.uncertainty[1] == 0.0
    np.testing.assert_array_equal(result.accepted, [False, True])


@pytest.mark.parametrize("notion", ["EU", "eu", EpistemicUncertainty])
def test_notion_accepts_aliases_and_classes(notion: Any) -> None:  # noqa: ANN401
    predictor = SelectivePredictor(_RepresentationModel(_ensemble_sample()), ThresholdSelector(0.2), notion=notion)

    assert predictor.notion is EpistemicUncertainty


def test_default_notion_is_total() -> None:
    assert SelectivePredictor(_model(), ThresholdSelector(0.2)).notion is TotalUncertainty


def test_invalid_notion_raises() -> None:
    with pytest.raises(ValueError, match=r"notion must be one of .*'tu', 'TU', got 'bogus'"):
        SelectivePredictor(_model(), ThresholdSelector(0.2), notion="bogus")


def test_class_that_is_not_a_notion_raises() -> None:
    with pytest.raises(TypeError, match="subclass of Notion"):
        SelectivePredictor(_model(), ThresholdSelector(0.2), notion=int)


def test_callable_notion_computes_criterion_from_representation() -> None:
    model = _model()
    seen = []

    def one_minus_max_probability(representation: Any) -> np.ndarray:  # noqa: ANN401
        seen.append(representation)
        return 1.0 - representation.probabilities.max(axis=-1)

    result = SelectivePredictor(model, ThresholdSelector(0.3), notion=one_minus_max_probability).predict(None)

    assert seen == [model.representation]
    np.testing.assert_allclose(result.uncertainty, 1.0 - PROBABILITIES.max(axis=-1))
    np.testing.assert_array_equal(result.accepted, [True, False, False, True])
    np.testing.assert_allclose(
        result.decision.probabilities,
        categorical_from_mean(model.representation).probabilities,
    )


def test_callable_notion_skips_quantify(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_representation: object) -> None:
        msg = "quantify must not be called for a callable notion"
        raise AssertionError(msg)

    monkeypatch.setattr("probly.selective_prediction._common.quantify", fail)
    result = SelectivePredictor(
        _model(), ThresholdSelector(0.5), notion=lambda _rep: np.array([0.1, 0.9, 0.5, 0.6])
    ).predict(None)

    np.testing.assert_array_equal(result.accepted, [True, False, True, False])


def test_non_callable_notion_raises() -> None:
    with pytest.raises(TypeError, match="notion"):
        SelectivePredictor(_model(), ThresholdSelector(0.2), notion=0.5)


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


def _random_ensemble_sample() -> NumpyCategoricalDistributionSample:
    probabilities = np.random.default_rng(0).dirichlet([1.0, 1.0, 1.0], size=(5, 50))
    return NumpyCategoricalDistributionSample(
        array=NumpyProbabilityCategoricalDistribution(probabilities),
        sample_axis=0,
    )


@pytest.mark.parametrize("cost", [0.1, 0.25, 0.4])
def test_threshold_on_default_criterion_is_chows_rule(cost: float) -> None:
    sample = _random_ensemble_sample()
    for representation, probabilities in [
        (sample, sample.sample_mean().probabilities),
        (create_categorical_distribution(PROBABILITIES), PROBABILITIES),
    ]:
        result = SelectivePredictor(_RepresentationModel(representation), ThresholdSelector(cost)).predict(None)

        np.testing.assert_array_equal(result.accepted, probabilities.max(axis=-1) >= 1.0 - cost)


def test_default_decomposition_is_additive() -> None:
    sample = _random_ensemble_sample()
    components = {
        notion: SelectivePredictor(_RepresentationModel(sample), ThresholdSelector(0.5), notion=notion)
        .predict(None)
        .uncertainty
        for notion in ("total", "aleatoric", "epistemic")
    }

    np.testing.assert_allclose(components["total"], components["aleatoric"] + components["epistemic"])
    assert bool((components["epistemic"] >= 0).all())


def test_default_epistemic_is_exactly_zero_where_members_agree() -> None:
    # Computed as total minus aleatoric, the epistemic part of these instances is float noise of about 1e-16,
    # partly negative, so a threshold at zero would accept an arbitrary subset of them.
    probabilities = np.random.default_rng(0).dirichlet([5.0, 1.0, 1.0], size=(5, 200))
    probabilities = probabilities[:, (probabilities.argmax(axis=-1) == 0).all(axis=0)]
    sample = NumpyCategoricalDistributionSample(
        array=NumpyProbabilityCategoricalDistribution(probabilities),
        sample_axis=0,
    )

    result = SelectivePredictor(_RepresentationModel(sample), ThresholdSelector(0.0), notion="epistemic").predict(None)

    np.testing.assert_array_equal(result.uncertainty, 0.0)
    assert bool(result.accepted.all())


@pytest.mark.parametrize("notion", ["total", "aleatoric", "epistemic"])
def test_log_loss_gives_entropy_decomposition(notion: str) -> None:
    sample = _random_ensemble_sample()
    result = SelectivePredictor(
        _RepresentationModel(sample), ThresholdSelector(0.5), notion=notion, loss=LogLoss()
    ).predict(None)

    np.testing.assert_allclose(result.uncertainty, quantify(sample)[notion])


def test_log_loss_gives_entropy_of_single_distribution() -> None:
    result = SelectivePredictor(_model(), ThresholdSelector(0.5), loss=LogLoss()).predict(None)

    np.testing.assert_allclose(result.uncertainty, ENTROPIES)


def test_brier_loss_gives_its_scoring_rule_decomposition() -> None:
    sample = _random_ensemble_sample()
    result = SelectivePredictor(
        _RepresentationModel(sample), ThresholdSelector(0.5), notion="epistemic", loss=BrierLoss()
    ).predict(None)

    np.testing.assert_allclose(result.uncertainty, SecondOrderScoringRuleDecomposition(sample, BrierLoss()).epistemic)


def test_no_loss_uses_quantify() -> None:
    sample = _random_ensemble_sample()
    result = SelectivePredictor(
        _RepresentationModel(sample), ThresholdSelector(0.5), notion="epistemic", loss=None
    ).predict(None)

    np.testing.assert_allclose(result.uncertainty, quantify(sample).epistemic)


def test_dirichlet_criterion_uses_zero_one_decomposition() -> None:
    alphas = np.array([[8.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1.0, 5.0, 0.5]])
    dirichlet = create_dirichlet_distribution_from_alphas(alphas)
    expected = SecondOrderZeroOneDecomposition(dirichlet)

    total = SelectivePredictor(_RepresentationModel(dirichlet), ThresholdSelector(0.5)).predict(None)
    np.testing.assert_allclose(total.uncertainty, 1.0 - (alphas / alphas.sum(-1, keepdims=True)).max(-1))
    np.testing.assert_allclose(total.uncertainty, expected.total)


def test_dirichlet_rejects_other_losses() -> None:
    dirichlet = create_dirichlet_distribution_from_alphas(np.array([[2.0, 1.0], [1.0, 1.0]]))
    predictor = SelectivePredictor(_RepresentationModel(dirichlet), ThresholdSelector(0.5), loss=LogLoss())

    with pytest.raises(NotImplementedError, match="LogLoss is not supported for Dirichlet"):
        predictor.predict(None)


def test_default_loss_is_zero_one() -> None:
    assert isinstance(SelectivePredictor(_model(), ThresholdSelector(0.2)).loss, ZeroOneLoss)


@pytest.mark.parametrize("loss", [ZeroOneLoss(), LogLoss(), BrierLoss()])
def test_explicit_loss_raises_for_representation_it_does_not_apply_to(loss: Any) -> None:  # noqa: ANN401
    predictor = SelectivePredictor(_RepresentationModel(_regression_sample()), ThresholdSelector(0.5), loss=loss)

    with pytest.raises(NotImplementedError, match=f"{type(loss).__name__} is not supported for"):
        predictor.predict(None)


def test_explicit_zero_one_loss_matches_default_where_it_applies() -> None:
    sample = _random_ensemble_sample()
    default = SelectivePredictor(_RepresentationModel(sample), ThresholdSelector(0.5), notion="epistemic")
    explicit = SelectivePredictor(
        _RepresentationModel(sample), ThresholdSelector(0.5), notion="epistemic", loss=ZeroOneLoss()
    )

    np.testing.assert_array_equal(explicit.predict(None).uncertainty, default.predict(None).uncertainty)


def test_invalid_loss_raises() -> None:
    with pytest.raises(TypeError, match="loss"):
        SelectivePredictor(_model(), ThresholdSelector(0.2), loss="zero_one")


def test_callable_notion_ignores_loss() -> None:
    result = SelectivePredictor(
        _model(), ThresholdSelector(0.5), notion=lambda rep: 1.0 - rep.probabilities.max(axis=-1), loss=LogLoss()
    ).predict(None)

    np.testing.assert_allclose(result.uncertainty, MAX_PROB_COMPLEMENTS)


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
    np.testing.assert_allclose(sp.predict(None).uncertainty, MAX_PROB_COMPLEMENTS)


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


def _kth_smallest(values: np.ndarray, coverage: float) -> float:
    n = len(values)
    k = int(np.ceil((n + 1) * coverage))
    return float(np.sort(values)[k - 1])


@pytest.mark.parametrize(("n", "coverage"), [(19, 0.9), (50, 0.8), (99, 0.5), (200, 0.95), (10, 0.75)])
def test_coverage_threshold_is_order_statistic(n: int, coverage: float) -> None:
    kappa = np.random.default_rng(n).permutation(n).astype(float) / n
    selector = CoverageSelector(coverage).calibrate(kappa)
    assert selector.threshold == _kth_smallest(kappa, coverage)
    assert isinstance(selector.threshold, float)
    accepted = selector.select(kappa)
    assert accepted.sum() == int(np.ceil((n + 1) * coverage))


@pytest.mark.parametrize(("n", "coverage"), [(5, 0.9), (9, 0.95), (3, 1.0), (10, 1.0)])
def test_coverage_threshold_is_inf_if_rank_exceeds_n(n: int, coverage: float) -> None:
    kappa = np.arange(n, dtype=float)
    selector = CoverageSelector(coverage).calibrate(kappa)
    assert selector.threshold == np.inf
    assert selector.select(np.array([1e9, np.nan])).tolist() == [True, False]


@pytest.mark.parametrize("coverage", [0.0, -0.1, 1.5, float("nan")])
def test_coverage_must_be_in_unit_interval(coverage: float) -> None:
    with pytest.raises(ValueError, match="coverage"):
        CoverageSelector(coverage)


def test_coverage_selector_requires_calibration_and_returns_self() -> None:
    selector = CoverageSelector(0.9)
    assert selector.threshold is None
    with pytest.raises(ValueError, match="not calibrated"):
        selector.select(np.array([0.1]))
    assert selector.calibrate(np.linspace(0.0, 1.0, 20)) is selector


@pytest.mark.parametrize(
    "bad",
    [np.array([]), np.array([0.1, np.nan]), np.zeros((2, 2))],
)
def test_coverage_calibration_rejects_invalid_arrays(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="uncertainty"):
        CoverageSelector(0.9).calibrate(bad)


def test_coverage_calibration_rejects_non_array() -> None:
    with pytest.raises(TypeError, match="array"):
        CoverageSelector(0.9).calibrate([0.1, 0.2])


def test_coverage_ties_at_threshold_are_accepted() -> None:
    kappa = np.array([0.1] * 10 + [0.5] * 10)
    selector = CoverageSelector(0.6).calibrate(kappa)
    assert selector.threshold == 0.5
    assert selector.select(kappa).all()


def test_coverage_guarantee_by_simulation() -> None:
    rng = np.random.default_rng(0)
    n, coverage, repeats = 19, 0.9, 20000
    realized = np.empty(repeats)
    for i in range(repeats):
        cal, test = rng.random(n), rng.random()
        realized[i] = CoverageSelector(coverage).calibrate(cal).select(np.array([test]))[0]
    # Without ties the expected coverage is exactly ceil((n + 1) * c) / (n + 1) = 18 / 20.
    se = np.sqrt(0.9 * 0.1 / repeats)
    assert coverage - 4 * se <= realized.mean() <= coverage + 1 / (n + 1) + 4 * se


def test_coverage_selector_is_a_calibrator() -> None:
    selector = CoverageSelector(0.9)
    assert isinstance(selector, Calibrator)
    assert calibrate(selector, np.linspace(0.0, 1.0, 20)) is selector
    assert selector.threshold is not None


def test_pipeline_calibrate_matches_selector_on_predicted_criterion() -> None:
    sp = SelectivePredictor(_model(), CoverageSelector(0.5))
    assert sp.calibrate(None) is sp
    expected = CoverageSelector(0.5).calibrate(1.0 - PROBABILITIES.max(axis=-1))
    assert sp.selector.threshold == expected.threshold
    result = sp.predict(None)
    np.testing.assert_array_equal(result.accepted, result.uncertainty <= expected.threshold)


def test_pipeline_calibrate_uses_callable_notion() -> None:
    sp = SelectivePredictor(_model(), CoverageSelector(0.5), notion=lambda _rep: np.array([4.0, 3.0, 2.0, 1.0]))
    sp.calibrate(None)
    assert sp.selector.threshold == 3.0


def test_pipeline_calibrate_requires_fitted_selector() -> None:
    with pytest.raises(TypeError, match="ThresholdSelector"):
        SelectivePredictor(_model(), ThresholdSelector(0.5)).calibrate(None)
