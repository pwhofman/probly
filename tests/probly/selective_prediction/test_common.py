"""Backend-agnostic tests for selective predictors, run with NumPy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

import numpy as np
import pytest
from scipy.stats import binom

from probly.calibrator import Calibrator, calibrate
from probly.decider import categorical_from_mean
from probly.evaluation.selective_prediction import evaluate_selective_prediction
from probly.method import cast
from probly.predictor import RepresentationPredictor, predict
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
from probly.selective_prediction import SelectivePrediction, SelectivePredictor, Selector, selector as make_selector
from probly.selective_prediction._common import _binomial_upper_bound

if TYPE_CHECKING:
    from collections.abc import Callable

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
    result = SelectivePredictor(model, selector=make_selector(threshold=0.5)).predict(None)
    representation = model.predict_representation(None)

    assert isinstance(result, SelectivePrediction)
    np.testing.assert_allclose(result.uncertainty, MAX_PROB_COMPLEMENTS)
    np.testing.assert_allclose(
        result.decision.probabilities,
        categorical_from_mean(representation).probabilities,
    )


def test_accepted_is_uncertainty_at_most_threshold() -> None:
    result = SelectivePredictor(_model(), selector=make_selector(threshold=0.3)).predict(None)

    np.testing.assert_array_equal(result.accepted, result.uncertainty <= 0.3)
    np.testing.assert_array_equal(result.accepted, [True, False, False, True])
    assert result.coverage == 0.5


def test_tie_at_threshold_is_accepted() -> None:
    result = SelectivePredictor(_model(), selector=make_selector(threshold=float(MAX_PROB_COMPLEMENTS[2]))).predict(
        None
    )

    assert result.accepted[2]


@pytest.mark.parametrize(("threshold", "coverage"), [(np.inf, 1.0), (-np.inf, 0.0)])
def test_infinite_thresholds_accept_all_or_none(threshold: float, coverage: float) -> None:
    result = SelectivePredictor(_model(), selector=make_selector(threshold=threshold)).predict(None)

    assert result.coverage == coverage


def test_coverage_of_empty_batch_is_nan() -> None:
    result = SelectivePrediction(decision=None, uncertainty=np.zeros(0), accepted=np.zeros(0, dtype=bool))

    assert np.isnan(result.coverage)


def test_nan_threshold_raises() -> None:
    with pytest.raises(ValueError, match="NaN"):
        make_selector(threshold=float("nan"))


def test_threshold_selector_works_on_plain_arrays() -> None:
    selector = make_selector(threshold=0.5)
    uncertainty = np.array([0.1, 0.5, 0.7, np.nan])

    np.testing.assert_array_equal(selector.select(uncertainty), [True, True, False, False])


@pytest.mark.parametrize("uncertainty", [np.float64(0.1), np.zeros((2, 2))])
def test_threshold_selector_rejects_non_one_dimensional_uncertainty(uncertainty: np.ndarray) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        make_selector(threshold=0.5).select(uncertainty)


def test_threshold_selector_rejects_non_array_uncertainty() -> None:
    with pytest.raises(TypeError, match="array"):
        make_selector(threshold=0.5).select([0.1, 0.9])


def test_predictor_rejects_criterion_with_extra_axes() -> None:
    predictor = SelectivePredictor(
        _model(), selector=make_selector(threshold=0.5), notion=lambda rep: rep.probabilities
    )

    with pytest.raises(ValueError, match=r"shape \(4, 2\)"):
        predictor.predict(None)


@pytest.mark.parametrize("notion", ["total", "aleatoric", "epistemic"])
def test_notion_selects_component_of_decomposition(notion: str) -> None:
    sample = _ensemble_sample()
    predictor = SelectivePredictor(_RepresentationModel(sample), selector=make_selector(threshold=0.2), notion=notion)

    result = predictor.predict(None)

    np.testing.assert_allclose(result.uncertainty, SecondOrderZeroOneDecomposition(sample)[notion])


def test_epistemic_notion_separates_disagreement_from_noise() -> None:
    result = SelectivePredictor(
        _RepresentationModel(_ensemble_sample()), selector=make_selector(threshold=0.1), notion="epistemic"
    ).predict(None)

    assert result.uncertainty[0] > 0.1
    assert result.uncertainty[1] == 0.0
    np.testing.assert_array_equal(result.accepted, [False, True])


@pytest.mark.parametrize("notion", ["EU", "eu", EpistemicUncertainty])
def test_notion_accepts_aliases_and_classes(notion: Any) -> None:  # noqa: ANN401
    predictor = SelectivePredictor(
        _RepresentationModel(_ensemble_sample()), selector=make_selector(threshold=0.2), notion=notion
    )

    assert predictor.notion is EpistemicUncertainty


def test_default_notion_is_total() -> None:
    assert SelectivePredictor(_model(), selector=make_selector(threshold=0.2)).notion is TotalUncertainty


def test_invalid_notion_raises() -> None:
    with pytest.raises(ValueError, match=r"notion must be one of .*'tu', 'TU', got 'bogus'"):
        SelectivePredictor(_model(), selector=make_selector(threshold=0.2), notion="bogus")


def test_class_that_is_not_a_notion_raises() -> None:
    with pytest.raises(TypeError, match="subclass of Notion"):
        SelectivePredictor(_model(), selector=make_selector(threshold=0.2), notion=int)


def test_callable_notion_computes_criterion_from_representation() -> None:
    model = _model()
    seen = []

    def one_minus_max_probability(representation: Any) -> np.ndarray:  # noqa: ANN401
        seen.append(representation)
        return 1.0 - representation.probabilities.max(axis=-1)

    result = SelectivePredictor(model, selector=make_selector(threshold=0.3), notion=one_minus_max_probability).predict(
        None
    )

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
        _model(), selector=make_selector(threshold=0.5), notion=lambda _rep: np.array([0.1, 0.9, 0.5, 0.6])
    ).predict(None)

    np.testing.assert_array_equal(result.accepted, [True, False, True, False])


def test_quantify_result_that_is_not_a_decomposition_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("probly.selective_prediction._common.quantify", lambda _representation: 0.5)
    predictor = SelectivePredictor(_model(), selector=make_selector(threshold=0.2), loss=None)

    with pytest.raises(TypeError, match="Expected quantify to return a Decomposition"):
        predictor.predict(None)


def test_non_callable_notion_raises() -> None:
    with pytest.raises(TypeError, match="notion"):
        SelectivePredictor(_model(), selector=make_selector(threshold=0.2), notion=0.5)


def test_notion_missing_from_decomposition_raises() -> None:
    predictor = SelectivePredictor(_model(), selector=make_selector(threshold=0.2), notion="epistemic")

    with pytest.raises(KeyError, match=r"Probability\w* has no EpistemicUncertainty .* only: TotalUncertainty"):
        predictor.predict(None)


@pytest.mark.parametrize("notion", ["total", "aleatoric", "epistemic"])
def test_regression_criterion_comes_from_quantify(notion: str) -> None:
    sample = _regression_sample()
    result = SelectivePredictor(
        _RepresentationModel(sample), selector=make_selector(threshold=1.0), notion=notion, decider=lambda rep: rep
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
        result = SelectivePredictor(
            _RepresentationModel(representation), selector=make_selector(threshold=cost)
        ).predict(None)

        np.testing.assert_array_equal(result.accepted, probabilities.max(axis=-1) >= 1.0 - cost)


def test_default_decomposition_is_additive() -> None:
    sample = _random_ensemble_sample()
    components = {
        notion: SelectivePredictor(_RepresentationModel(sample), selector=make_selector(threshold=0.5), notion=notion)
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

    result = SelectivePredictor(
        _RepresentationModel(sample), selector=make_selector(threshold=0.0), notion="epistemic"
    ).predict(None)

    np.testing.assert_array_equal(result.uncertainty, 0.0)
    assert bool(result.accepted.all())


@pytest.mark.parametrize("notion", ["total", "aleatoric", "epistemic"])
def test_log_loss_gives_entropy_decomposition(notion: str) -> None:
    sample = _random_ensemble_sample()
    result = SelectivePredictor(
        _RepresentationModel(sample), selector=make_selector(threshold=0.5), notion=notion, loss=LogLoss()
    ).predict(None)

    np.testing.assert_allclose(result.uncertainty, quantify(sample)[notion])


def test_log_loss_gives_entropy_of_single_distribution() -> None:
    result = SelectivePredictor(_model(), selector=make_selector(threshold=0.5), loss=LogLoss()).predict(None)

    np.testing.assert_allclose(result.uncertainty, ENTROPIES)


def test_brier_loss_gives_its_scoring_rule_decomposition() -> None:
    sample = _random_ensemble_sample()
    result = SelectivePredictor(
        _RepresentationModel(sample), selector=make_selector(threshold=0.5), notion="epistemic", loss=BrierLoss()
    ).predict(None)

    np.testing.assert_allclose(result.uncertainty, SecondOrderScoringRuleDecomposition(sample, BrierLoss()).epistemic)


def test_no_loss_uses_quantify() -> None:
    sample = _random_ensemble_sample()
    result = SelectivePredictor(
        _RepresentationModel(sample), selector=make_selector(threshold=0.5), notion="epistemic", loss=None
    ).predict(None)

    np.testing.assert_allclose(result.uncertainty, quantify(sample).epistemic)


def test_dirichlet_criterion_uses_zero_one_decomposition() -> None:
    alphas = np.array([[8.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1.0, 5.0, 0.5]])
    dirichlet = create_dirichlet_distribution_from_alphas(alphas)
    expected = SecondOrderZeroOneDecomposition(dirichlet)

    total = SelectivePredictor(_RepresentationModel(dirichlet), selector=make_selector(threshold=0.5)).predict(None)
    np.testing.assert_allclose(total.uncertainty, 1.0 - (alphas / alphas.sum(-1, keepdims=True)).max(-1))
    np.testing.assert_allclose(total.uncertainty, expected.total)


def test_dirichlet_rejects_other_losses() -> None:
    dirichlet = create_dirichlet_distribution_from_alphas(np.array([[2.0, 1.0], [1.0, 1.0]]))
    predictor = SelectivePredictor(
        _RepresentationModel(dirichlet), selector=make_selector(threshold=0.5), loss=LogLoss()
    )

    with pytest.raises(NotImplementedError, match="LogLoss is not supported for Dirichlet"):
        predictor.predict(None)


def test_default_loss_is_zero_one() -> None:
    assert isinstance(SelectivePredictor(_model(), selector=make_selector(threshold=0.2)).loss, ZeroOneLoss)


@pytest.mark.parametrize("loss", [ZeroOneLoss(), LogLoss(), BrierLoss()])
def test_explicit_loss_raises_for_representation_it_does_not_apply_to(loss: Any) -> None:  # noqa: ANN401
    predictor = SelectivePredictor(
        _RepresentationModel(_regression_sample()), selector=make_selector(threshold=0.5), loss=loss
    )

    with pytest.raises(NotImplementedError, match=f"{type(loss).__name__} is not supported for"):
        predictor.predict(None)


def test_explicit_zero_one_loss_matches_default_where_it_applies() -> None:
    sample = _random_ensemble_sample()
    default = SelectivePredictor(
        _RepresentationModel(sample), selector=make_selector(threshold=0.5), notion="epistemic"
    )
    explicit = SelectivePredictor(
        _RepresentationModel(sample), selector=make_selector(threshold=0.5), notion="epistemic", loss=ZeroOneLoss()
    )

    np.testing.assert_array_equal(explicit.predict(None).uncertainty, default.predict(None).uncertainty)


def test_invalid_loss_raises() -> None:
    with pytest.raises(TypeError, match="loss"):
        SelectivePredictor(_model(), selector=make_selector(threshold=0.2), loss="zero_one")


def test_callable_notion_ignores_loss() -> None:
    result = SelectivePredictor(
        _model(),
        selector=make_selector(threshold=0.5),
        notion=lambda rep: 1.0 - rep.probabilities.max(axis=-1),
        loss=LogLoss(),
    ).predict(None)

    np.testing.assert_allclose(result.uncertainty, MAX_PROB_COMPLEMENTS)


def test_custom_decider_is_used() -> None:
    result = SelectivePredictor(
        _model(), selector=make_selector(threshold=0.2), decider=lambda rep: rep.probabilities.argmax(-1)
    ).predict(None)

    np.testing.assert_array_equal(result.decision, [0, 0, 0, 0])


def test_predict_dispatch_agrees_with_method() -> None:
    sp = SelectivePredictor(_model(), selector=make_selector(threshold=0.5))

    via_method = sp.predict(None)
    via_dispatch = predict(sp, None)

    np.testing.assert_array_equal(via_dispatch.accepted, via_method.accepted)
    np.testing.assert_allclose(via_dispatch.uncertainty, via_method.uncertainty)


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
    sp = SelectivePredictor(
        _SamplingModel(), selector=make_selector(threshold=0.5), representer_kwargs={"num_samples": 3}
    )

    assert isinstance(sp.representer, _SamplingRepresenter)
    assert sp.representer.num_samples == 3
    np.testing.assert_allclose(sp.predict(None).uncertainty, MAX_PROB_COMPLEMENTS)


def test_missing_representer_kwargs_raise() -> None:
    with pytest.raises(TypeError, match="num_samples"):
        SelectivePredictor(_SamplingModel(), selector=make_selector(threshold=0.5))


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


def test_subclass_overriding_only_select_is_a_selector() -> None:
    selector = _LeastUncertainSelector(k=1)

    np.testing.assert_array_equal(selector.select(MAX_PROB_COMPLEMENTS), [False, False, False, True])
    assert not selector.takes_losses


def test_subclass_without_threshold_works_in_the_predictor() -> None:
    result = SelectivePredictor(_model(), selector=_LeastUncertainSelector(k=1)).predict(None)
    np.testing.assert_array_equal(result.accepted, [False, False, False, True])


def _kth_smallest(values: np.ndarray, coverage: float) -> float:
    n = len(values)
    k = int(np.ceil((n + 1) * coverage))
    return float(np.sort(values)[k - 1])


@pytest.mark.parametrize(("n", "coverage"), [(19, 0.9), (50, 0.8), (99, 0.5), (200, 0.95), (10, 0.75)])
def test_coverage_threshold_is_order_statistic(n: int, coverage: float) -> None:
    kappa = np.random.default_rng(n).permutation(n).astype(float) / n
    selector = make_selector(coverage=coverage).calibrate(kappa)
    assert selector.threshold == _kth_smallest(kappa, coverage)
    assert isinstance(selector.threshold, float)
    accepted = selector.select(kappa)
    assert accepted.sum() == int(np.ceil((n + 1) * coverage))


def test_coverage_rank_matches_exact_arithmetic_on_a_grid() -> None:
    for n in range(1, 201):
        kappa = np.arange(n, dtype=float)
        for hundredths in range(1, 101):
            rank = -((-(n + 1) * hundredths) // 100)  # exact ceil((n + 1) * c) for c = hundredths / 100
            selector = make_selector(coverage=hundredths / 100)
            if rank > n and hundredths < 100:
                needed = -(-hundredths // (100 - hundredths))  # exact ceil(c / (1 - c))
                with pytest.raises(ValueError, match=f"at least {needed} are needed"):
                    selector.calibrate(kappa)
                assert needed > n, (n, hundredths)
                continue
            threshold = selector.calibrate(kappa).threshold
            assert threshold == (np.inf if rank > n else float(rank - 1)), (n, hundredths)


@pytest.mark.parametrize("n", [1, 3, 10])
def test_full_coverage_threshold_is_inf(n: int) -> None:
    selector = make_selector(coverage=1.0).calibrate(np.arange(n, dtype=float))
    assert selector.threshold == np.inf
    assert selector.select(np.array([1e9, np.nan])).tolist() == [True, False]


@pytest.mark.parametrize(("n", "coverage", "needed"), [(3, 0.8, 4), (8, 0.9, 9), (18, 0.95, 19)])
def test_coverage_raises_if_calibration_set_is_too_small(n: int, coverage: float, needed: int) -> None:
    selector = make_selector(coverage=coverage)
    with pytest.raises(ValueError, match=f"too few for coverage={coverage}: at least {needed} are needed"):
        selector.calibrate(np.arange(n, dtype=float))
    assert not selector.is_calibrated
    assert selector.calibrate(np.arange(needed, dtype=float)).threshold == needed - 1


@pytest.mark.parametrize("coverage", [0.0, -0.1, 1.5, float("nan")])
def test_coverage_must_be_in_unit_interval(coverage: float) -> None:
    with pytest.raises(ValueError, match="coverage"):
        make_selector(coverage=coverage)


def test_coverage_selector_requires_calibration_and_returns_self() -> None:
    selector = make_selector(coverage=0.9)
    assert selector.threshold is None
    assert not selector.is_calibrated
    with pytest.raises(ValueError, match="not calibrated"):
        selector.select(np.array([0.1]))
    assert selector.calibrate(np.linspace(0.0, 1.0, 20)) is selector
    assert selector.is_calibrated


@pytest.mark.parametrize(
    "bad",
    [np.array([]), np.array([0.1, np.nan]), np.zeros((2, 2))],
)
def test_coverage_calibration_rejects_invalid_arrays(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="uncertainty"):
        make_selector(coverage=0.9).calibrate(bad)


def test_coverage_calibration_rejects_non_array() -> None:
    with pytest.raises(TypeError, match="array"):
        make_selector(coverage=0.9).calibrate([0.1, 0.2])


def test_coverage_ties_at_threshold_are_accepted() -> None:
    kappa = np.array([0.1] * 10 + [0.5] * 10)
    selector = make_selector(coverage=0.6).calibrate(kappa)
    assert selector.threshold == 0.5
    assert selector.select(kappa).all()


def test_coverage_guarantee_by_simulation() -> None:
    rng = np.random.default_rng(0)
    n, coverage, repeats = 19, 0.9, 20000
    realized = np.empty(repeats)
    for i in range(repeats):
        cal, test = rng.random(n), rng.random()
        realized[i] = make_selector(coverage=coverage).calibrate(cal).select(np.array([test]))[0]
    # Without ties the expected coverage is exactly ceil((n + 1) * c) / (n + 1) = 18 / 20.
    se = np.sqrt(0.9 * 0.1 / repeats)
    assert coverage - 4 * se <= realized.mean() <= coverage + 1 / (n + 1) + 4 * se


def test_coverage_selector_is_a_calibrator() -> None:
    selector = make_selector(coverage=0.9)
    assert isinstance(selector, Calibrator)
    assert calibrate(selector, np.linspace(0.0, 1.0, 20)) is selector
    assert selector.threshold is not None


def test_pipeline_calibrate_matches_selector_on_predicted_criterion() -> None:
    sp = SelectivePredictor(_model(), selector=make_selector(coverage=0.5))
    assert sp.calibrate(None, None) is sp
    expected = make_selector(coverage=0.5).calibrate(1.0 - PROBABILITIES.max(axis=-1))
    assert sp.selector.threshold == expected.threshold
    result = sp.predict(None)
    np.testing.assert_array_equal(result.accepted, result.uncertainty <= expected.threshold)


def test_pipeline_calibrate_uses_callable_notion() -> None:
    sp = SelectivePredictor(
        _model(), selector=make_selector(coverage=0.5), notion=lambda _rep: np.array([4.0, 3.0, 2.0, 1.0])
    )
    sp.calibrate(None, None)
    assert sp.selector.threshold == 3.0


def test_pipeline_calibrate_requires_fitted_rule() -> None:
    with pytest.raises(TypeError, match=r"^selector\(threshold=0\.5\) is not fitted on data"):
        SelectivePredictor(_model(), selector=make_selector(threshold=0.5)).calibrate(None, None)


@pytest.mark.parametrize("errors", [0, 1, 4, 17])
def test_binomial_upper_bound_inverts_the_binomial_cdf(errors: int) -> None:
    n, delta = 40, 0.05
    bound = _binomial_upper_bound(errors, n, delta)
    assert binom.cdf(errors, n, bound) == pytest.approx(delta)


def test_binomial_upper_bound_edge_cases() -> None:
    assert _binomial_upper_bound(10, 10, 0.1) == 1.0
    assert _binomial_upper_bound(0, 10, 0.1) == pytest.approx(1.0 - 0.1 ** (1 / 10))


def _reference_sgr(kappa: np.ndarray, losses: np.ndarray, risk: float, delta: float) -> tuple[float, float, list[int]]:
    """Plain transcription of Algorithm 1 of Geifman and El-Yaniv (2017), returning the last certified iterate."""
    n = len(kappa)
    ordered = np.sort(kappa)
    steps = int(np.ceil(np.log2(n)))
    z_min, z_max = 1, n
    certified = (-np.inf, np.nan)
    tested = []
    for _ in range(steps):
        z = int(np.ceil((z_min + z_max) / 2))
        tau = ordered[n - z]
        mask = kappa <= tau
        tested.append(int(mask.sum()))
        bound = _binomial_upper_bound(losses[mask].sum(), int(mask.sum()), delta / steps)
        if bound < risk:
            z_max = z
            certified = (tau, bound)
        else:
            z_min = z
    return certified[0], certified[1], tested


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("ties", [False, True])
def test_sgr_matches_reference_transcription(seed: int, ties: bool) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(50, 400))
    kappa = rng.random(n)
    if ties:
        kappa = np.round(kappa, 1)
    losses = (rng.random(n) < 0.02 + 0.2 * kappa**2).astype(float)
    expected_threshold, expected_bound, _ = _reference_sgr(kappa, losses, 0.15, 0.2)
    selector = make_selector(risk=0.15, delta=0.2).calibrate(kappa, losses)
    assert selector.threshold == expected_threshold
    np.testing.assert_allclose(selector.bound, expected_bound)


def test_sgr_hand_made_case() -> None:
    kappa = np.arange(8) / 10
    losses = np.zeros(8)
    # Tested accepted counts: 4, 6, 7 (all certified), so the threshold is the 7th smallest criterion.
    assert _reference_sgr(kappa, losses, 0.5, 0.5)[2] == [4, 6, 7]
    selector = make_selector(risk=0.5, delta=0.5).calibrate(kappa, losses)
    assert selector.threshold == pytest.approx(0.6)
    assert selector.bound == pytest.approx(1.0 - (0.5 / 3) ** (1 / 7))
    np.testing.assert_array_equal(selector.select(kappa), kappa <= 0.6)


def test_sgr_returns_last_certified_threshold_not_last_tested() -> None:
    kappa = np.arange(16) / 16
    losses = np.zeros(16)
    losses[12:] = 1.0
    selector = make_selector(risk=0.3, delta=0.5).calibrate(kappa, losses)
    assert selector.threshold is not None
    assert selector.bound is not None
    assert selector.bound < 0.3


@pytest.mark.parametrize(("n", "risk"), [(1, 0.5), (2, 0.9), (50, 0.01)])
def test_sgr_raises_without_certifiable_threshold(n: int, risk: float) -> None:
    kappa = np.linspace(0.0, 1.0, n)
    selector = make_selector(risk=risk, delta=0.1)
    with pytest.raises(ValueError, match="No threshold could be certified"):
        selector.calibrate(kappa, np.ones(n))
    assert not selector.is_calibrated
    assert selector.bound is None


def test_sgr_failed_calibration_keeps_the_previous_fit() -> None:
    kappa = np.linspace(0.0, 1.0, 50)
    selector = make_selector(risk=0.5, delta=0.1).calibrate(kappa, np.zeros(50))
    fitted = (selector.threshold, selector.bound)
    with pytest.raises(ValueError, match="No threshold could be certified"):
        selector.calibrate(kappa, np.ones(50))
    assert (selector.threshold, selector.bound) == fitted


@pytest.mark.parametrize(("risk", "delta"), [(0.0, 0.1), (1.0, 0.1), (-0.1, 0.1), (0.1, 0.0), (0.1, 1.0)])
def test_sgr_risk_and_delta_must_be_in_open_unit_interval(risk: float, delta: float) -> None:
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        make_selector(risk=risk, delta=delta)


def test_sgr_requires_losses() -> None:
    with pytest.raises(TypeError, match="losses"):
        make_selector(risk=0.1, delta=0.1).calibrate(np.linspace(0.0, 1.0, 20))


@pytest.mark.parametrize(
    ("kappa", "losses", "error", "match"),
    [
        (np.zeros(5), np.zeros(4), ValueError, "same length"),
        (np.zeros(0), np.zeros(0), ValueError, "at least one"),
        (np.array([0.1, np.nan]), np.zeros(2), ValueError, "NaN"),
        (np.zeros(2), np.array([0.0, np.nan]), ValueError, "NaN"),
        (np.zeros(2), np.array([0.0, 0.5]), ValueError, "binary"),
        (np.zeros(2), np.array([0.0, 2.0]), ValueError, "binary"),
        (np.zeros((2, 2)), np.zeros(2), ValueError, "^uncertainty must be one-dimensional"),
        (np.zeros(2), np.zeros((2, 2)), ValueError, "^losses must be one-dimensional"),
        ([0.1, 0.2], np.zeros(2), TypeError, "^uncertainty must be an array"),
        (np.zeros(2), [0.0, 0.0], TypeError, "^losses must be an array"),
    ],
)
def test_sgr_calibration_rejects_invalid_input(kappa: Any, losses: Any, error: type[Exception], match: str) -> None:  # noqa: ANN401
    with pytest.raises(error, match=match):
        make_selector(risk=0.1, delta=0.1).calibrate(kappa, losses)


def test_sgr_requires_calibration_before_select() -> None:
    selector = make_selector(risk=0.1, delta=0.1)
    assert selector.threshold is None
    assert selector.bound is None
    assert not selector.is_calibrated
    with pytest.raises(ValueError, match="not calibrated"):
        selector.select(np.zeros(3))


def test_sgr_selector_is_a_calibrator() -> None:
    selector = make_selector(risk=0.2, delta=0.1)
    assert isinstance(selector, Calibrator)
    rng = np.random.default_rng(0)
    kappa = rng.random(500)
    assert calibrate(selector, kappa, np.zeros(500)) is selector
    assert selector.is_calibrated
    assert selector.threshold is not None
    assert selector.threshold > 0.0


def test_sgr_risk_guarantee_by_simulation() -> None:
    rng = np.random.default_rng(0)
    n, risk, delta, repeats = 500, 0.05, 0.2, 300
    # The error probability is 0.3 * kappa^2, so the selective risk at threshold t is 0.1 * t^2, at most 0.05 up to
    # t = sqrt(0.5).
    limit = np.sqrt(0.5)
    violations = 0
    for _ in range(repeats):
        kappa = rng.random(n)
        losses = (rng.random(n) < 0.3 * kappa**2).astype(float)
        try:
            threshold = make_selector(risk=risk, delta=delta).calibrate(kappa, losses).threshold
        except ValueError:
            continue  # Nothing certified: no threshold is returned, so the guarantee cannot be violated.
        violations += threshold > limit
    assert violations / repeats <= delta


def test_pipeline_calibrate_with_labels_matches_selector_on_losses() -> None:
    labels = np.array([0, 1, 0, 0])
    sp = SelectivePredictor(_model(), selector=make_selector(risk=0.9, delta=0.5))
    assert sp.calibrate(labels, None) is sp
    expected = make_selector(risk=0.9, delta=0.5).calibrate(MAX_PROB_COMPLEMENTS, np.array([0.0, 1.0, 0.0, 0.0]))
    assert sp.selector.threshold == expected.threshold
    np.testing.assert_equal(sp.selector.bound, expected.bound)


def test_pipeline_losses_follow_the_decision_not_the_ground_truth_class_index() -> None:
    # The decision is class 0 for every instance, so a target of 1 is an error and a target of 0 is not.
    sp = SelectivePredictor(_model(), selector=make_selector(risk=0.99, delta=0.5))
    losses = sp._losses(sp.representer.represent(None), np.array([1, 1, 0, 0]))  # noqa: SLF001
    np.testing.assert_array_equal(losses, [1.0, 1.0, 0.0, 0.0])


class _CountingModel(_RepresentationModel):
    """Stub model that counts its forward passes."""

    calls = 0

    @override
    def predict_representation(self, _x: object) -> Any:
        self.calls += 1
        return self.representation


@pytest.mark.parametrize(
    ("selector", "labels", "match"),
    [
        (make_selector(coverage=0.5), np.zeros(4, dtype=int), r"^selector\(coverage=0\.5\) takes no labels"),
        (make_selector(risk=0.5, delta=0.5), None, r"^selector\(risk=.*\) needs the labels"),
    ],
)
def test_pipeline_label_mismatch_raises_before_the_forward_pass(
    selector: Selector,
    labels: np.ndarray | None,
    match: str,
) -> None:
    model = _CountingModel(create_categorical_distribution(PROBABILITIES))
    with pytest.raises(TypeError, match=match):
        SelectivePredictor(model, selector=selector).calibrate(labels, None)
    assert model.calls == 0


def test_pipeline_calibrate_without_model_input_raises() -> None:
    model = _CountingModel(create_categorical_distribution(PROBABILITIES))
    with pytest.raises(TypeError, match="No input of the model"):
        SelectivePredictor(model, selector=make_selector(coverage=0.5)).calibrate(None)
    assert model.calls == 0


def test_risk_rule_needs_a_task_loss_at_construction() -> None:
    with pytest.raises(TypeError, match=r"^selector\(risk=.*\) needs losses"):
        SelectivePredictor(_model(), selector=make_selector(risk=0.5, delta=0.5), loss=None)


def test_pipeline_non_binary_loss_is_rejected_by_sgr() -> None:
    sp = SelectivePredictor(_model(), selector=make_selector(risk=0.5, delta=0.5), loss=LogLoss())
    with pytest.raises(ValueError, match="binary"):
        sp.calibrate(np.zeros(4, dtype=int), None)


@pytest.mark.parametrize(
    ("labels", "match"),
    [
        (np.array([0, 1, 2, 0]), "in \\[0, 2\\)"),
        (np.array([0, -1, 0, 0]), "in \\[0, 2\\)"),
        (np.array([0.0, 0.5, 0.0, 0.0]), "integer"),
        (np.array([0, 1, 0]), "one class index"),
        (np.zeros((4, 1), dtype=int), "one class index"),
    ],
)
def test_pipeline_rejects_invalid_labels(labels: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        SelectivePredictor(_model(), selector=make_selector(risk=0.5, delta=0.5)).calibrate(labels, None)


def test_pipeline_scoring_rule_on_a_regression_decision_raises() -> None:
    sp = SelectivePredictor(
        _RepresentationModel(_regression_sample()), selector=make_selector(risk=0.5, delta=0.5), decider=lambda r: r
    )
    with pytest.raises(TypeError, match="A scoring rule needs a categorical decision, got NumpyGaussianDistribution"):
        sp.calibrate(np.zeros(3, dtype=int), None)


def test_pipeline_calibrate_through_the_calibrator_protocol_takes_labels_first() -> None:
    sp = SelectivePredictor(_model(), selector=make_selector(risk=0.9, delta=0.5))
    calibrate(sp, np.array([0, 1, 0, 0]), None)
    expected = make_selector(risk=0.9, delta=0.5).calibrate(MAX_PROB_COMPLEMENTS, np.array([0.0, 1.0, 0.0, 0.0]))
    assert sp.selector.threshold == expected.threshold


# The selector factory.


def test_selector_builds_the_selector_of_its_keywords() -> None:
    assert repr(make_selector(threshold=0.3)) == "selector(threshold=0.3)"
    assert repr(make_selector(coverage=0.8)) == "selector(coverage=0.8)"
    assert repr(make_selector(risk=0.1, delta=0.2)) == "selector(risk=0.1, delta=0.2)"
    assert make_selector(threshold=0.3, coverage=None).threshold == 0.3


@pytest.mark.parametrize(
    ("rules", "match"),
    [
        ({}, "exactly one rule"),
        ({"threshold": 0.5, "coverage": 0.5}, "exactly one rule"),
        ({"coverage": 0.5, "risk": 0.1, "delta": 0.1}, "exactly one rule"),
        ({"delta": 0.1}, "given together"),
        ({"risk": 0.1}, "given together"),
    ],
)
def test_selector_keywords_must_form_one_rule(rules: dict[str, float], match: str) -> None:
    with pytest.raises(TypeError, match=match):
        make_selector(**rules)


@pytest.mark.parametrize(
    "rules",
    [
        {"threshold": float("nan")},
        {"coverage": 0.0},
        {"coverage": 1.5},
        {"risk": 1.0, "delta": 0.1},
        {"risk": 0.1, "delta": 0.0},
    ],
)
def test_selector_keywords_out_of_range_raise(rules: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=r"NaN|coverage|\(0, 1\)"):
        make_selector(**rules)


# The selector argument.


def test_selector_must_be_a_selector() -> None:
    with pytest.raises(TypeError, match="selector must be a Selector or None"):
        SelectivePredictor(_model(), selector=0.5)


def test_no_rule_is_allowed_but_cannot_predict_or_calibrate() -> None:
    sp = SelectivePredictor(_model())

    assert sp.selector is None
    with pytest.raises(TypeError, match="No selector to select with"):
        sp.predict(None)
    with pytest.raises(TypeError, match="no selector to calibrate"):
        sp.calibrate(None, None)


@pytest.mark.parametrize("method", ["predict", "evaluate"])
@pytest.mark.parametrize("selector", [make_selector(coverage=0.5), make_selector(risk=0.5, delta=0.5)])
def test_uncalibrated_rule_raises_before_the_forward_pass(selector: Selector, method: str) -> None:
    model = _CountingModel(create_categorical_distribution(PROBABILITIES))
    sp = SelectivePredictor(model, selector=selector)
    arguments = (None,) if method == "predict" else (np.zeros(4, dtype=int), None)

    with pytest.raises(ValueError, match=r"^selector\(.*\) is not calibrated\. Call calibrate\(\) first\."):
        getattr(sp, method)(*arguments)
    assert model.calls == 0


class _UncalibratedSelector(Selector):
    """Custom fitted selector that reports itself as not calibrated."""

    is_calibrated = False

    @override
    def __repr__(self) -> str:
        return "selector(custom)"

    @override
    def select(self, uncertainty: Any) -> np.ndarray:
        msg = "select must not be reached."
        raise AssertionError(msg)


def test_custom_uncalibrated_selector_raises_before_the_forward_pass() -> None:
    model = _CountingModel(create_categorical_distribution(PROBABILITIES))
    sp = SelectivePredictor(model, selector=_UncalibratedSelector())

    with pytest.raises(ValueError, match=r"^selector\(custom\) is not calibrated\."):
        sp.predict(None)
    assert model.calls == 0


def test_reassigned_selector_that_needs_losses_meets_loss_none_in_calibrate() -> None:
    sp = SelectivePredictor(_model(), loss=None)
    sp.selector = make_selector(risk=0.5, delta=0.5)

    with pytest.raises(TypeError, match=r"needs losses, but loss is None"):
        sp.calibrate(np.zeros(4, dtype=int), None)


def test_criterion_with_extra_axes_hint_names_the_predictor_type() -> None:
    predictor = SelectivePredictor(
        _model(),
        selector=make_selector(threshold=0.5),
        notion=lambda _representation: np.zeros((4, 4)),
    )

    with pytest.raises(ValueError, match="one-dimensional") as info:
        predictor.predict(None)
    assert any("predictor_type" in note for note in info.value.__notes__)


# Introspection.


def test_selector_threshold_is_fixed_or_fitted() -> None:
    assert SelectivePredictor(_model(), selector=make_selector(threshold=0.3)).selector.threshold == 0.3
    sp = SelectivePredictor(_model(), selector=make_selector(coverage=0.5))
    assert sp.selector.threshold is None
    sp.calibrate(None, None)
    assert sp.selector.threshold is not None


def test_selector_bound_belongs_to_the_risk_rule() -> None:
    assert SelectivePredictor(_model(), selector=make_selector(threshold=0.3)).selector.bound is None
    assert (
        SelectivePredictor(_model(), selector=make_selector(coverage=0.5)).calibrate(None, None).selector.bound is None
    )
    sp = SelectivePredictor(_model(), selector=make_selector(risk=0.9, delta=0.5))
    assert sp.selector.bound is None
    sp.calibrate(np.array([0, 1, 0, 0]), None)
    expected = make_selector(risk=0.9, delta=0.5).calibrate(MAX_PROB_COMPLEMENTS, np.array([0.0, 1.0, 0.0, 0.0]))
    np.testing.assert_equal(sp.selector.bound, expected.bound)


def test_pipeline_calibrate_raises_if_nothing_was_certified() -> None:
    sp = SelectivePredictor(_model(), selector=make_selector(risk=0.01, delta=0.1))
    with pytest.raises(ValueError, match="No threshold could be certified"):
        sp.calibrate(np.ones(4, dtype=int), None)
    assert not sp.selector.is_calibrated
    assert sp.selector.bound is None


# Models.


class _ProbaModel:
    """Stub scikit-learn-like classifier."""

    def predict_proba(self, _x: object) -> np.ndarray:
        return PROBABILITIES


class _Plain:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x


@pytest.mark.parametrize(
    "model",
    [
        _model(),
        _SamplingModel(),
        _ProbaModel(),
        cast(lambda r: r, predictor_type=RepresentationPredictor),
    ],
)
def test_representable_models_are_wrapped(model: Any) -> None:  # noqa: ANN401
    kwargs: dict[str, Any] = {"representer_kwargs": {"num_samples": 2}} if isinstance(model, _SamplingModel) else {}

    assert isinstance(SelectivePredictor(model, **kwargs).representer, Representer)


@pytest.mark.parametrize(
    "model",
    [lambda x: x, _Plain(), _ProbaModel().predict_proba, object()],
    ids=["function", "callable object", "bound method", "object"],
)
def test_model_that_probly_cannot_represent_is_rejected_at_construction(model: Any) -> None:  # noqa: ANN401
    with pytest.raises(TypeError, match=r"is not a model that probly can represent.*apply a selector to it"):
        SelectivePredictor(model, selector=make_selector(coverage=0.5))


def test_evaluate_without_loss_suggests_a_loss() -> None:
    sp = SelectivePredictor(_model(), loss=None)

    with pytest.raises(TypeError, match=r"^evaluate needs losses\. Pass a scoring rule or a function loss\.$"):
        sp.evaluate(np.zeros(4, dtype=int), None)


def test_criterion_that_is_not_an_array_is_rejected_by_predict_and_evaluate() -> None:
    sp = SelectivePredictor(
        _model(),
        selector=make_selector(threshold=0.3),
        notion=lambda rep: list(1 - rep.probabilities.max(-1)),
    )

    with pytest.raises(TypeError, match="must be an array"):
        sp.predict(None)
    with pytest.raises(TypeError, match="must be an array"):
        sp.evaluate(np.zeros(4, dtype=int), None)


def test_evaluate_label_errors_do_not_name_calibration() -> None:
    sp = SelectivePredictor(_model())

    with pytest.raises(ValueError, match=r"^The labels must be in"):
        sp.evaluate(np.array([0, 1, 0, 5]), None)


# Function losses.


def _cost(decision: Any, y: np.ndarray) -> np.ndarray:  # noqa: ANN401
    return np.where(decision.probabilities.argmax(axis=-1) == y, 0.0, 1.0)


def test_function_loss_keeps_the_zero_one_criterion() -> None:
    # quantify would decompose a sample of categorical distributions by entropy instead.
    sample = _random_ensemble_sample()
    sp = SelectivePredictor(
        _RepresentationModel(sample), selector=make_selector(threshold=0.5), notion="epistemic", loss=_cost
    )

    result = sp.predict(None)

    np.testing.assert_allclose(result.uncertainty, SecondOrderZeroOneDecomposition(sample).epistemic)


def test_function_loss_receives_the_decision_and_the_labels() -> None:
    labels = np.array([0, 1, 0, 0])
    seen = []

    def loss(decision: Any, y: np.ndarray) -> np.ndarray:  # noqa: ANN401
        seen.append((decision, y))
        return _cost(decision, y)

    sp = SelectivePredictor(
        _model(), selector=make_selector(risk=0.9, delta=0.5), loss=loss, decider=categorical_from_mean
    )
    sp.calibrate(labels, None)

    ((decision, y),) = seen
    assert y is labels
    np.testing.assert_allclose(decision.probabilities, PROBABILITIES)
    expected = make_selector(risk=0.9, delta=0.5).calibrate(MAX_PROB_COMPLEMENTS, np.array([0.0, 1.0, 0.0, 0.0]))
    assert sp.selector.threshold == expected.threshold


def test_function_loss_must_return_one_value_per_instance() -> None:
    sp = SelectivePredictor(_model(), loss=lambda d, _y: d.probabilities)

    with pytest.raises(ValueError, match=r"one value per instance, got shape \(4, 2\)"):
        sp.evaluate(np.zeros(4, dtype=int), None)


def test_regression_with_a_function_loss_uses_quantify_and_evaluates() -> None:
    sample = _regression_sample()
    y = np.array([0.0, 1.0, 2.0])
    sp = SelectivePredictor(
        _RepresentationModel(sample),
        selector=make_selector(threshold=1.0),
        notion="epistemic",
        decider=lambda s: s.array.mean.mean(axis=0),
        loss=lambda d, y: (d - y) ** 2,
    )

    result = sp.evaluate(y, None)

    criterion = quantify(sample).epistemic
    losses = (sample.array.mean.mean(axis=0) - y) ** 2
    np.testing.assert_allclose(sp.predict(None).uncertainty, criterion)
    assert result["risk@1"] == pytest.approx(losses.mean())
    assert "accuracy@1" not in result
    accepted = criterion <= 1.0
    assert result["selected:coverage"] == pytest.approx(accepted.mean())
    assert result["selected:risk"] == pytest.approx(losses[accepted].mean())


# Notes on errors in user functions.


def _raise_value_error(*_arguments: object) -> None:
    msg = "user error"
    raise ValueError(msg)


@pytest.mark.parametrize(
    ("make", "call", "note"),
    [
        (
            lambda: SelectivePredictor(_model(), selector=make_selector(threshold=0.5), notion=_raise_value_error),
            lambda sp: sp.predict(None),
            "notion receives the representation (NumpyProbabilityCategoricalDistribution), not the model output.",
        ),
        (
            lambda: SelectivePredictor(_model(), selector=make_selector(threshold=0.5), decider=_raise_value_error),
            lambda sp: sp.predict(None),
            "decider receives the representation",
        ),
        (
            lambda: SelectivePredictor(_model(), selector=make_selector(threshold=0.5), loss=_raise_value_error),
            lambda sp: sp.evaluate(np.zeros(4, dtype=int), None),
            "loss receives the decision of the decider and the labels",
        ),
    ],
)
def test_errors_in_user_functions_keep_their_type_and_get_a_note(
    make: Callable[[], SelectivePredictor],
    call: Callable[[SelectivePredictor], object],
    note: str,
) -> None:
    sp = make()
    with pytest.raises(ValueError, match="user error") as info:
        call(sp)
    assert len(info.value.__notes__) == 1
    assert note in info.value.__notes__[0]


# Evaluation.

_LABELS = np.array([0, 1, 0, 1])  # The decision is class 0 everywhere, so instances 1 and 3 are errors.
_ZERO_ONE_LOSSES = np.array([0.0, 1.0, 0.0, 1.0])


def test_evaluate_needs_model_input() -> None:
    sp = SelectivePredictor(_model(), selector=make_selector(threshold=0.3))

    with pytest.raises(TypeError, match=r"No input of the model was given.*evaluate\(y, x\)"):
        sp.evaluate(_LABELS)


def test_evaluate_matches_the_metrics_on_criterion_and_losses() -> None:
    result = SelectivePredictor(_model(), selector=make_selector(threshold=0.3)).evaluate(_LABELS, None)

    expected = evaluate_selective_prediction(MAX_PROB_COMPLEMENTS, _ZERO_ONE_LOSSES)
    accepted = MAX_PROB_COMPLEMENTS <= 0.3
    assert set(result) == {*expected, "selected:coverage", "selected:risk"}
    for key, value in expected.items():
        assert result[key] == pytest.approx(value)
    assert result["selected:coverage"] == pytest.approx(accepted.mean())
    assert result["selected:risk"] == pytest.approx(_ZERO_ONE_LOSSES[accepted].mean())


def test_evaluate_runs_the_model_once() -> None:
    model = _CountingModel(create_categorical_distribution(PROBABILITIES))

    SelectivePredictor(model, selector=make_selector(threshold=0.3)).evaluate(_LABELS, None)

    assert model.calls == 1


def test_evaluate_without_a_rule_reports_only_ranking_metrics() -> None:
    result = SelectivePredictor(_model(), notion=lambda _r: np.array([0.1, 0.2, 0.3, 0.4])).evaluate(_LABELS, None)

    assert not any(key.startswith("selected:") for key in result)


def test_evaluate_passes_metrics_on() -> None:
    result = SelectivePredictor(_model()).evaluate(_LABELS, None, metrics=["aurc"])

    assert set(result) == {"aurc"}
    assert result["aurc"] == pytest.approx(
        evaluate_selective_prediction(MAX_PROB_COMPLEMENTS, _ZERO_ONE_LOSSES)["aurc"]
    )


def test_evaluate_with_a_calibrated_rule() -> None:
    sp = SelectivePredictor(_model(), selector=make_selector(coverage=0.5)).calibrate(None, None)

    result = sp.evaluate(_LABELS, None)

    assert result["selected:coverage"] == pytest.approx(sp.predict(None).coverage)


def test_evaluate_maps_nan_criterion_to_inf_for_metrics_only() -> None:
    criterion = np.array([0.1, np.nan, 0.3, 0.2])
    sp = SelectivePredictor(_model(), selector=make_selector(threshold=np.inf), notion=lambda _r: criterion)

    result = sp.evaluate(_LABELS, None)

    expected = evaluate_selective_prediction(np.array([0.1, np.inf, 0.3, 0.2]), _ZERO_ONE_LOSSES)
    for key, value in expected.items():
        assert result[key] == pytest.approx(value)
    # The rule rejects the NaN criterion although an infinite threshold accepts +inf.
    assert result["selected:coverage"] == pytest.approx(0.75)
    assert result["selected:risk"] == pytest.approx(1 / 3)


def test_evaluate_risk_is_nan_if_nothing_is_accepted() -> None:
    result = SelectivePredictor(_model(), selector=make_selector(threshold=-np.inf)).evaluate(_LABELS, None)

    assert result["selected:coverage"] == 0.0
    assert np.isnan(result["selected:risk"])


def test_evaluate_needs_losses_before_the_forward_pass() -> None:
    model = _CountingModel(create_categorical_distribution(PROBABILITIES))

    with pytest.raises(TypeError, match="evaluate needs losses"):
        SelectivePredictor(model, selector=make_selector(threshold=0.5), loss=None).evaluate(_LABELS, None)
    assert model.calls == 0


def test_evaluate_rejects_losses_of_another_length() -> None:
    sp = SelectivePredictor(_model(), selector=make_selector(threshold=0.5), loss=lambda _d, y: y)

    with pytest.raises(ValueError, match=r"same shape, got \(4,\) and \(3,\)"):
        sp.evaluate(np.zeros(3), None)


# Batching recipe.


def test_concatenated_representations_on_an_identity_model_match_one_batch() -> None:
    representation = create_categorical_distribution(PROBABILITIES)
    concatenated = np.concatenate([representation[:2], representation[2:]], axis=0)
    identity = cast(lambda r: r, predictor_type=RepresentationPredictor)

    sp = SelectivePredictor(identity, selector=make_selector(risk=0.9, delta=0.5)).calibrate(
        np.array([0, 1, 0, 0]), concatenated
    )
    reference = SelectivePredictor(_model(), selector=make_selector(risk=0.9, delta=0.5)).calibrate(
        np.array([0, 1, 0, 0]), None
    )

    assert type(concatenated) is type(representation)
    assert sp.selector.threshold == reference.selector.threshold
