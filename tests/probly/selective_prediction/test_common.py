"""Backend-agnostic tests for selective predictors, run with NumPy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

import numpy as np
import pytest

from probly.calibrator import calibrate
from probly.decider import categorical_from_mean
from probly.evaluation.selective_prediction import evaluate_selective_prediction
from probly.method import cast
from probly.metrics import brier_loss_per_instance, log_loss_per_instance, zero_one_loss_per_instance
from probly.predictor import RepresentationPredictor, predict
from probly.quantification import (
    EpistemicUncertainty,
    SecondOrderScoringRuleDecomposition,
    SecondOrderZeroOneDecomposition,
    TotalUncertainty,
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
    SelectivePrediction,
    SelectivePredictor,
    Selector,
    selective_predictor,
    selector as make_selector,
)
from probly.selective_prediction._common import _losses

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
    result = selective_predictor(representer(model), selector=make_selector(threshold=0.5)).predict(None)
    representation = model.predict_representation(None)

    assert isinstance(result, SelectivePrediction)
    np.testing.assert_allclose(result.uncertainty, MAX_PROB_COMPLEMENTS)
    np.testing.assert_allclose(
        result.decision.probabilities,
        categorical_from_mean(representation).probabilities,
    )


def test_accepted_is_uncertainty_at_most_threshold() -> None:
    result = selective_predictor(representer(_model()), selector=make_selector(threshold=0.3)).predict(None)

    np.testing.assert_array_equal(result.accepted, result.uncertainty <= 0.3)
    np.testing.assert_array_equal(result.accepted, [True, False, False, True])
    assert result.coverage == 0.5


def test_tie_at_threshold_is_accepted() -> None:
    result = selective_predictor(
        representer(_model()), selector=make_selector(threshold=float(MAX_PROB_COMPLEMENTS[2]))
    ).predict(None)

    assert result.accepted[2]


@pytest.mark.parametrize(("threshold", "coverage"), [(np.inf, 1.0), (-np.inf, 0.0)])
def test_infinite_thresholds_accept_all_or_none(threshold: float, coverage: float) -> None:
    result = selective_predictor(representer(_model()), selector=make_selector(threshold=threshold)).predict(None)

    assert result.coverage == coverage


def test_coverage_of_empty_batch_is_nan() -> None:
    result = SelectivePrediction(decision=None, uncertainty=np.zeros(0), accepted=np.zeros(0, dtype=bool))

    assert np.isnan(result.coverage)


def test_predictor_rejects_criterion_with_extra_axes() -> None:
    predictor = selective_predictor(
        representer(_model()), selector=make_selector(threshold=0.5, notion=lambda rep: rep.probabilities)
    )

    with pytest.raises(ValueError, match=r"shape \(4, 2\)"):
        predictor.predict(None)


@pytest.mark.parametrize("notion", ["total", "aleatoric", "epistemic"])
def test_notion_selects_component_of_decomposition(notion: str) -> None:
    sample = _ensemble_sample()
    predictor = selective_predictor(
        representer(_RepresentationModel(sample)), selector=make_selector(threshold=0.2, notion=notion)
    )

    result = predictor.predict(None)

    np.testing.assert_allclose(result.uncertainty, SecondOrderZeroOneDecomposition(sample)[notion])


def test_epistemic_notion_separates_disagreement_from_noise() -> None:
    result = selective_predictor(
        representer(_RepresentationModel(_ensemble_sample())), selector=make_selector(threshold=0.1, notion="epistemic")
    ).predict(None)

    assert result.uncertainty[0] > 0.1
    assert result.uncertainty[1] == 0.0
    np.testing.assert_array_equal(result.accepted, [False, True])


@pytest.mark.parametrize("notion", ["EU", "eu", EpistemicUncertainty])
def test_notion_accepts_aliases_and_classes(notion: Any) -> None:  # noqa: ANN401
    predictor = selective_predictor(
        representer(_RepresentationModel(_ensemble_sample())), selector=make_selector(threshold=0.2, notion=notion)
    )

    assert predictor.notion is EpistemicUncertainty


def test_default_notion_is_total() -> None:
    assert selective_predictor(representer(_model()), selector=make_selector(threshold=0.2)).notion is TotalUncertainty


def test_notion_is_read_from_the_selector() -> None:
    rule = make_selector(threshold=0.2)
    predictor = selective_predictor(representer(_model()), selector=rule)
    rule.notion = EpistemicUncertainty

    assert predictor.notion is EpistemicUncertainty


def test_notion_without_selector_is_total() -> None:
    assert selective_predictor(representer(_model())).notion is TotalUncertainty


def test_invalid_notion_raises() -> None:
    with pytest.raises(ValueError, match=r"notion must be one of .*'tu', 'TU', got 'bogus'"):
        selective_predictor(representer(_model()), selector=make_selector(threshold=0.2, notion="bogus"))


def test_class_that_is_not_a_notion_raises() -> None:
    with pytest.raises(TypeError, match="subclass of Notion"):
        selective_predictor(representer(_model()), selector=make_selector(threshold=0.2, notion=int))


def test_callable_notion_computes_criterion_from_representation() -> None:
    model = _model()
    seen = []

    def one_minus_max_probability(representation: Any) -> np.ndarray:  # noqa: ANN401
        seen.append(representation)
        return 1.0 - representation.probabilities.max(axis=-1)

    result = selective_predictor(
        representer(model), selector=make_selector(threshold=0.3, notion=one_minus_max_probability)
    ).predict(None)

    assert seen == [model.representation]
    np.testing.assert_allclose(result.uncertainty, 1.0 - PROBABILITIES.max(axis=-1))
    np.testing.assert_array_equal(result.accepted, [True, False, False, True])
    np.testing.assert_allclose(
        result.decision.probabilities,
        categorical_from_mean(model.representation).probabilities,
    )


def test_callable_notion_skips_the_decomposition(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_representation: object, _loss: object) -> None:
        msg = "scoring_rule_decomposition must not be called for a callable notion"
        raise AssertionError(msg)

    monkeypatch.setattr("probly.selective_prediction._common.scoring_rule_decomposition", fail)
    result = selective_predictor(
        representer(_model()), selector=make_selector(threshold=0.5, notion=lambda _rep: np.array([0.1, 0.9, 0.5, 0.6]))
    ).predict(None)

    np.testing.assert_array_equal(result.accepted, [True, False, True, False])


def test_non_callable_notion_raises() -> None:
    with pytest.raises(TypeError, match="notion"):
        selective_predictor(representer(_model()), selector=make_selector(threshold=0.2, notion=0.5))


def test_notion_missing_from_decomposition_raises() -> None:
    predictor = selective_predictor(representer(_model()), selector=make_selector(threshold=0.2, notion="epistemic"))

    with pytest.raises(KeyError, match=r"Probability\w* has no EpistemicUncertainty .* only: TotalUncertainty"):
        predictor.predict(None)


@pytest.mark.parametrize(
    "kwargs", [{}, {"loss": None}, {"loss": lambda d, y: (d - y) ** 2}], ids=["default", "none", "function"]
)
def test_named_notion_without_a_scoring_rule_decomposition_raises(kwargs: dict[str, Any]) -> None:
    predictor = selective_predictor(
        representer(_RepresentationModel(_regression_sample())),
        selector=make_selector(threshold=1.0),
        decider=lambda rep: rep,
        **kwargs,
    )

    with pytest.raises(
        NotImplementedError, match=r"zero_one_loss_per_instance is not supported for NumpyGaussianDistributionSample"
    ):
        predictor.predict(None)


@pytest.mark.parametrize("notion", ["total", "aleatoric", "epistemic"])
def test_regression_criterion_from_a_quantify_notion(notion: str) -> None:
    sample = _regression_sample()
    result = selective_predictor(
        representer(_RepresentationModel(sample)),
        selector=make_selector(threshold=1.0, notion=lambda rep: quantify(rep)[notion]),
        decider=lambda rep: rep,
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
        result = selective_predictor(
            representer(_RepresentationModel(representation)), selector=make_selector(threshold=cost)
        ).predict(None)

        np.testing.assert_array_equal(result.accepted, probabilities.max(axis=-1) >= 1.0 - cost)


def test_default_decomposition_is_additive() -> None:
    sample = _random_ensemble_sample()
    components = {
        notion: selective_predictor(
            representer(_RepresentationModel(sample)), selector=make_selector(threshold=0.5, notion=notion)
        )
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

    result = selective_predictor(
        representer(_RepresentationModel(sample)), selector=make_selector(threshold=0.0, notion="epistemic")
    ).predict(None)

    np.testing.assert_array_equal(result.uncertainty, 0.0)
    assert bool(result.accepted.all())


@pytest.mark.parametrize("notion", ["total", "aleatoric", "epistemic"])
def test_log_loss_gives_entropy_decomposition(notion: str) -> None:
    sample = _random_ensemble_sample()
    result = selective_predictor(
        representer(_RepresentationModel(sample)),
        selector=make_selector(threshold=0.5, notion=notion),
        loss=log_loss_per_instance,
    ).predict(None)

    np.testing.assert_allclose(result.uncertainty, quantify(sample)[notion])


def test_log_loss_gives_entropy_of_single_distribution() -> None:
    result = selective_predictor(
        representer(_model()), selector=make_selector(threshold=0.5), loss=log_loss_per_instance
    ).predict(None)

    np.testing.assert_allclose(result.uncertainty, ENTROPIES)


def test_brier_loss_gives_its_scoring_rule_decomposition() -> None:
    sample = _random_ensemble_sample()
    result = selective_predictor(
        representer(_RepresentationModel(sample)),
        selector=make_selector(threshold=0.5, notion="epistemic"),
        loss=brier_loss_per_instance,
    ).predict(None)

    np.testing.assert_allclose(
        result.uncertainty, SecondOrderScoringRuleDecomposition(sample, brier_loss_per_instance).epistemic
    )


def test_no_loss_uses_the_zero_one_decomposition() -> None:
    sample = _random_ensemble_sample()
    result = selective_predictor(
        representer(_RepresentationModel(sample)), selector=make_selector(threshold=0.5, notion="epistemic"), loss=None
    ).predict(None)

    np.testing.assert_allclose(result.uncertainty, SecondOrderZeroOneDecomposition(sample).epistemic)


def test_dirichlet_criterion_uses_zero_one_decomposition() -> None:
    alphas = np.array([[8.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1.0, 5.0, 0.5]])
    dirichlet = create_dirichlet_distribution_from_alphas(alphas)
    expected = SecondOrderZeroOneDecomposition(dirichlet)

    total = selective_predictor(
        representer(_RepresentationModel(dirichlet)), selector=make_selector(threshold=0.5)
    ).predict(None)
    np.testing.assert_allclose(total.uncertainty, 1.0 - (alphas / alphas.sum(-1, keepdims=True)).max(-1))
    np.testing.assert_allclose(total.uncertainty, expected.total)


def test_dirichlet_rejects_other_losses() -> None:
    dirichlet = create_dirichlet_distribution_from_alphas(np.array([[2.0, 1.0], [1.0, 1.0]]))
    predictor = selective_predictor(
        representer(_RepresentationModel(dirichlet)), selector=make_selector(threshold=0.5), loss=log_loss_per_instance
    )

    with pytest.raises(NotImplementedError, match="log_loss_per_instance is not supported for Dirichlet"):
        predictor.predict(None)


def test_default_loss_is_zero_one() -> None:
    assert (
        selective_predictor(representer(_model()), selector=make_selector(threshold=0.2)).loss
        is zero_one_loss_per_instance
    )


@pytest.mark.parametrize("loss", [zero_one_loss_per_instance, log_loss_per_instance, brier_loss_per_instance])
def test_explicit_loss_raises_for_representation_it_does_not_apply_to(loss: Any) -> None:  # noqa: ANN401
    predictor = selective_predictor(
        representer(_RepresentationModel(_regression_sample())), selector=make_selector(threshold=0.5), loss=loss
    )

    with pytest.raises(NotImplementedError, match=f"{loss.__name__} is not supported for"):
        predictor.predict(None)


def test_explicit_zero_one_loss_matches_default_where_it_applies() -> None:
    sample = _random_ensemble_sample()
    default = selective_predictor(
        representer(_RepresentationModel(sample)), selector=make_selector(threshold=0.5, notion="epistemic")
    )
    explicit = selective_predictor(
        representer(_RepresentationModel(sample)),
        selector=make_selector(threshold=0.5, notion="epistemic"),
        loss=zero_one_loss_per_instance,
    )

    np.testing.assert_array_equal(explicit.predict(None).uncertainty, default.predict(None).uncertainty)


def test_invalid_loss_raises() -> None:
    with pytest.raises(TypeError, match="loss"):
        selective_predictor(representer(_model()), selector=make_selector(threshold=0.2), loss="zero_one")


def test_callable_notion_ignores_loss() -> None:
    result = selective_predictor(
        representer(_model()),
        selector=make_selector(threshold=0.5, notion=lambda rep: 1.0 - rep.probabilities.max(axis=-1)),
        loss=log_loss_per_instance,
    ).predict(None)

    np.testing.assert_allclose(result.uncertainty, MAX_PROB_COMPLEMENTS)


def test_custom_decider_is_used() -> None:
    result = selective_predictor(
        representer(_model()), selector=make_selector(threshold=0.2), decider=lambda rep: rep.probabilities.argmax(-1)
    ).predict(None)

    np.testing.assert_array_equal(result.decision, [0, 0, 0, 0])


def test_predict_dispatch_agrees_with_method() -> None:
    sp = selective_predictor(representer(_model()), selector=make_selector(threshold=0.5))

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


def test_configured_representer_is_used_as_is() -> None:
    sampling_representer = representer(_SamplingModel(), num_samples=3)
    sp = selective_predictor(sampling_representer, selector=make_selector(threshold=0.5))

    assert sp.representer is sampling_representer
    assert sp.representer.num_samples == 3
    np.testing.assert_allclose(sp.predict(None).uncertainty, MAX_PROB_COMPLEMENTS)


class _LeastUncertainSelector(Selector):
    """Selector that overrides only `select` and keeps the k least uncertain instances."""

    def __init__(self, k: int) -> None:
        self.k = k

    @override
    def select(self, uncertainty: Any) -> np.ndarray:
        accepted = np.zeros(uncertainty.shape, dtype=bool)
        accepted[np.argsort(uncertainty, kind="stable")[: self.k]] = True
        return accepted


def test_subclass_without_threshold_works_in_the_predictor() -> None:
    result = selective_predictor(representer(_model()), selector=_LeastUncertainSelector(k=1)).predict(None)
    np.testing.assert_array_equal(result.accepted, [False, False, False, True])


def test_pipeline_calibrate_matches_selector_on_predicted_criterion() -> None:
    sp = selective_predictor(representer(_model()), selector=make_selector(coverage=0.5))
    assert sp.calibrate(None, None) is sp
    expected = make_selector(coverage=0.5).calibrate(1.0 - PROBABILITIES.max(axis=-1))
    assert sp.selector.threshold == expected.threshold
    result = sp.predict(None)
    np.testing.assert_array_equal(result.accepted, result.uncertainty <= expected.threshold)


def test_pipeline_calibrate_uses_callable_notion() -> None:
    sp = selective_predictor(
        representer(_model()), selector=make_selector(coverage=0.5, notion=lambda _rep: np.array([4.0, 3.0, 2.0, 1.0]))
    )
    sp.calibrate(None, None)
    assert sp.selector.threshold == 3.0


def test_pipeline_calibrate_requires_fitted_rule() -> None:
    with pytest.raises(TypeError, match=r"^selector\(threshold=0\.5\) is not fitted on data"):
        selective_predictor(representer(_model()), selector=make_selector(threshold=0.5)).calibrate(None, None)


def test_pipeline_calibrate_with_labels_matches_selector_on_losses() -> None:
    labels = np.array([0, 1, 0, 0])
    sp = selective_predictor(representer(_model()), selector=make_selector(risk=0.9, delta=0.5))
    assert sp.calibrate(labels, None) is sp
    expected = make_selector(risk=0.9, delta=0.5).calibrate(MAX_PROB_COMPLEMENTS, np.array([0.0, 1.0, 0.0, 0.0]))
    assert sp.selector.threshold == expected.threshold
    np.testing.assert_equal(sp.selector.bound, expected.bound)


def test_pipeline_losses_follow_the_decision_not_the_ground_truth_class_index() -> None:
    # The decision is class 0 for every instance, so a target of 1 is an error and a target of 0 is not.
    sp = selective_predictor(representer(_model()), selector=make_selector(risk=0.99, delta=0.5))
    losses = _losses(sp, sp.representer.represent(None), np.array([1, 1, 0, 0]))
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
        selective_predictor(representer(model), selector=selector).calibrate(labels, None)
    assert model.calls == 0


def test_pipeline_calibrate_without_input_raises() -> None:
    model = _CountingModel(create_categorical_distribution(PROBABILITIES))
    with pytest.raises(TypeError, match="No input of the representer"):
        selective_predictor(representer(model), selector=make_selector(coverage=0.5)).calibrate(None)
    assert model.calls == 0


def test_risk_rule_needs_a_task_loss_at_construction() -> None:
    with pytest.raises(TypeError, match=r"^selector\(risk=.*\) needs losses"):
        selective_predictor(representer(_model()), selector=make_selector(risk=0.5, delta=0.5), loss=None)


def test_pipeline_non_binary_loss_is_rejected_by_sgr() -> None:
    sp = selective_predictor(
        representer(_model()), selector=make_selector(risk=0.5, delta=0.5), loss=log_loss_per_instance
    )
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
        selective_predictor(representer(_model()), selector=make_selector(risk=0.5, delta=0.5)).calibrate(labels, None)


def test_pipeline_scoring_rule_on_a_regression_decision_raises() -> None:
    sp = selective_predictor(
        representer(_RepresentationModel(_regression_sample())),
        selector=make_selector(risk=0.5, delta=0.5, notion=lambda r: quantify(r).total),
        decider=lambda r: r,
    )
    with pytest.raises(TypeError, match="A scoring rule needs a categorical decision, got NumpyGaussianDistribution"):
        sp.calibrate(np.zeros(3, dtype=int), None)


def test_pipeline_calibrate_through_the_calibrator_protocol_takes_labels_first() -> None:
    sp = selective_predictor(representer(_model()), selector=make_selector(risk=0.9, delta=0.5))
    calibrate(sp, np.array([0, 1, 0, 0]), None)
    expected = make_selector(risk=0.9, delta=0.5).calibrate(MAX_PROB_COMPLEMENTS, np.array([0.0, 1.0, 0.0, 0.0]))
    assert sp.selector.threshold == expected.threshold


# The selector factory.


# The selector argument.


def test_selector_must_be_a_selector() -> None:
    with pytest.raises(TypeError, match="selector must be a Selector or None"):
        selective_predictor(representer(_model()), selector=0.5)


def test_no_rule_is_allowed_but_cannot_predict_or_calibrate() -> None:
    sp = selective_predictor(representer(_model()))

    assert sp.selector is None
    with pytest.raises(TypeError, match="No selector to select with"):
        sp.predict(None)
    with pytest.raises(TypeError, match="no selector to calibrate"):
        sp.calibrate(None, None)


@pytest.mark.parametrize("method", ["predict", "evaluate"])
@pytest.mark.parametrize("selector", [make_selector(coverage=0.5), make_selector(risk=0.5, delta=0.5)])
def test_uncalibrated_rule_raises_before_the_forward_pass(selector: Selector, method: str) -> None:
    model = _CountingModel(create_categorical_distribution(PROBABILITIES))
    sp = selective_predictor(representer(model), selector=selector)
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
    sp = selective_predictor(representer(model), selector=_UncalibratedSelector())

    with pytest.raises(ValueError, match=r"^selector\(custom\) is not calibrated\."):
        sp.predict(None)
    assert model.calls == 0


def test_reassigned_selector_that_needs_losses_meets_loss_none_in_calibrate() -> None:
    sp = selective_predictor(representer(_model()), loss=None)
    sp.selector = make_selector(risk=0.5, delta=0.5)

    with pytest.raises(TypeError, match=r"needs losses, but loss is None"):
        sp.calibrate(np.zeros(4, dtype=int), None)


def test_criterion_with_extra_axes_hint_names_the_predictor_type() -> None:
    predictor = selective_predictor(
        representer(_model()),
        selector=make_selector(threshold=0.5, notion=lambda _representation: np.zeros((4, 4))),
    )

    with pytest.raises(ValueError, match="one-dimensional") as info:
        predictor.predict(None)
    assert any("predictor_type" in note for note in info.value.__notes__)


# Introspection.


def test_selector_threshold_is_fixed_or_fitted() -> None:
    assert selective_predictor(representer(_model()), selector=make_selector(threshold=0.3)).selector.threshold == 0.3
    sp = selective_predictor(representer(_model()), selector=make_selector(coverage=0.5))
    assert sp.selector.threshold is None
    sp.calibrate(None, None)
    assert sp.selector.threshold is not None


def test_selector_bound_belongs_to_the_risk_rule() -> None:
    assert selective_predictor(representer(_model()), selector=make_selector(threshold=0.3)).selector.bound is None
    assert (
        selective_predictor(representer(_model()), selector=make_selector(coverage=0.5))
        .calibrate(None, None)
        .selector.bound
        is None
    )
    sp = selective_predictor(representer(_model()), selector=make_selector(risk=0.9, delta=0.5))
    assert sp.selector.bound is None
    sp.calibrate(np.array([0, 1, 0, 0]), None)
    expected = make_selector(risk=0.9, delta=0.5).calibrate(MAX_PROB_COMPLEMENTS, np.array([0.0, 1.0, 0.0, 0.0]))
    np.testing.assert_equal(sp.selector.bound, expected.bound)


def test_pipeline_calibrate_raises_if_nothing_was_certified() -> None:
    sp = selective_predictor(representer(_model()), selector=make_selector(risk=0.01, delta=0.1))
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
def test_representer_of_a_representable_model_is_used_as_is(model: Any) -> None:  # noqa: ANN401
    kwargs: dict[str, Any] = {"num_samples": 2} if isinstance(model, _SamplingModel) else {}
    model_representer = representer(model, **kwargs)

    assert selective_predictor(model_representer).representer is model_representer


@pytest.mark.parametrize(
    "base",
    [_model(), create_categorical_distribution(PROBABILITIES), lambda x: x, object()],
    ids=["representable model", "representation", "function", "object"],
)
def test_base_that_is_not_a_representer_is_rejected_at_construction(base: Any) -> None:  # noqa: ANN401
    with pytest.raises(TypeError, match=r"^base must be a Representer, got .*representer\(model, \.\.\.\)"):
        selective_predictor(base, selector=make_selector(coverage=0.5))


def test_evaluate_without_loss_suggests_a_loss() -> None:
    sp = selective_predictor(representer(_model()), loss=None)

    with pytest.raises(TypeError, match=r"^evaluate needs losses\. Pass a scoring rule or a function loss\.$"):
        sp.evaluate(np.zeros(4, dtype=int), None)


def test_criterion_that_is_not_an_array_is_rejected_by_predict_and_evaluate() -> None:
    sp = selective_predictor(
        representer(_model()),
        selector=make_selector(threshold=0.3, notion=lambda rep: list(1 - rep.probabilities.max(-1))),
    )

    with pytest.raises(TypeError, match="must be an array"):
        sp.predict(None)
    with pytest.raises(TypeError, match="must be an array"):
        sp.evaluate(np.zeros(4, dtype=int), None)


def test_evaluate_label_errors_do_not_name_calibration() -> None:
    sp = selective_predictor(representer(_model()))

    with pytest.raises(ValueError, match=r"^The labels must be in"):
        sp.evaluate(np.array([0, 1, 0, 5]), None)


# Function losses.


def _cost(decision: Any, y: np.ndarray) -> np.ndarray:  # noqa: ANN401
    return np.where(decision.probabilities.argmax(axis=-1) == y, 0.0, 1.0)


def test_function_loss_keeps_the_zero_one_criterion() -> None:
    # quantify would decompose a sample of categorical distributions by entropy instead.
    sample = _random_ensemble_sample()
    sp = selective_predictor(
        representer(_RepresentationModel(sample)), selector=make_selector(threshold=0.5, notion="epistemic"), loss=_cost
    )

    result = sp.predict(None)

    np.testing.assert_allclose(result.uncertainty, SecondOrderZeroOneDecomposition(sample).epistemic)


def test_function_loss_receives_the_decision_and_the_labels() -> None:
    labels = np.array([0, 1, 0, 0])
    seen = []

    def loss(decision: Any, y: np.ndarray) -> np.ndarray:  # noqa: ANN401
        seen.append((decision, y))
        return _cost(decision, y)

    sp = selective_predictor(
        representer(_model()), selector=make_selector(risk=0.9, delta=0.5), loss=loss, decider=categorical_from_mean
    )
    sp.calibrate(labels, None)

    ((decision, y),) = seen
    assert y is labels
    np.testing.assert_allclose(decision.probabilities, PROBABILITIES)
    expected = make_selector(risk=0.9, delta=0.5).calibrate(MAX_PROB_COMPLEMENTS, np.array([0.0, 1.0, 0.0, 0.0]))
    assert sp.selector.threshold == expected.threshold


def test_function_loss_must_return_one_value_per_instance() -> None:
    sp = selective_predictor(representer(_model()), loss=lambda d, _y: d.probabilities)

    with pytest.raises(ValueError, match=r"one value per instance, got shape \(4, 2\)"):
        sp.evaluate(np.zeros(4, dtype=int), None)


def test_regression_with_a_function_loss_evaluates() -> None:
    sample = _regression_sample()
    y = np.array([0.0, 1.0, 2.0])
    sp = selective_predictor(
        representer(_RepresentationModel(sample)),
        selector=make_selector(threshold=1.0, notion=lambda s: quantify(s).epistemic),
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
            lambda: selective_predictor(
                representer(_model()), selector=make_selector(threshold=0.5, notion=_raise_value_error)
            ),
            lambda sp: sp.predict(None),
            "notion receives the representation (NumpyProbabilityCategoricalDistribution), not the model output.",
        ),
        (
            lambda: selective_predictor(
                representer(_model()), selector=make_selector(threshold=0.5), decider=_raise_value_error
            ),
            lambda sp: sp.predict(None),
            "decider receives the representation",
        ),
        (
            lambda: selective_predictor(
                representer(_model()), selector=make_selector(threshold=0.5), loss=_raise_value_error
            ),
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


def test_evaluate_needs_input() -> None:
    sp = selective_predictor(representer(_model()), selector=make_selector(threshold=0.3))

    with pytest.raises(TypeError, match=r"No input of the representer was given.*evaluate\(y, x\)"):
        sp.evaluate(_LABELS)


def test_evaluate_matches_the_metrics_on_criterion_and_losses() -> None:
    result = selective_predictor(representer(_model()), selector=make_selector(threshold=0.3)).evaluate(_LABELS, None)

    expected = evaluate_selective_prediction(MAX_PROB_COMPLEMENTS, _ZERO_ONE_LOSSES)
    accepted = MAX_PROB_COMPLEMENTS <= 0.3
    assert set(result) == {*expected, "selected:coverage", "selected:risk"}
    for key, value in expected.items():
        assert result[key] == pytest.approx(value)
    assert result["selected:coverage"] == pytest.approx(accepted.mean())
    assert result["selected:risk"] == pytest.approx(_ZERO_ONE_LOSSES[accepted].mean())


def test_evaluate_runs_the_model_once() -> None:
    model = _CountingModel(create_categorical_distribution(PROBABILITIES))

    selective_predictor(representer(model), selector=make_selector(threshold=0.3)).evaluate(_LABELS, None)

    assert model.calls == 1


def test_evaluate_without_a_rule_reports_only_ranking_metrics() -> None:
    result = selective_predictor(representer(_model())).evaluate(_LABELS, None)

    assert not any(key.startswith("selected:") for key in result)


def test_evaluate_passes_metrics_on() -> None:
    result = selective_predictor(representer(_model())).evaluate(_LABELS, None, metrics=["aurc"])

    assert set(result) == {"aurc"}
    assert result["aurc"] == pytest.approx(
        evaluate_selective_prediction(MAX_PROB_COMPLEMENTS, _ZERO_ONE_LOSSES)["aurc"]
    )


def test_evaluate_with_a_calibrated_rule() -> None:
    sp = selective_predictor(representer(_model()), selector=make_selector(coverage=0.5)).calibrate(None, None)

    result = sp.evaluate(_LABELS, None)

    assert result["selected:coverage"] == pytest.approx(sp.predict(None).coverage)


def test_evaluate_maps_nan_criterion_to_inf_for_metrics_only() -> None:
    criterion = np.array([0.1, np.nan, 0.3, 0.2])
    sp = selective_predictor(
        representer(_model()), selector=make_selector(threshold=np.inf, notion=lambda _r: criterion)
    )

    result = sp.evaluate(_LABELS, None)

    expected = evaluate_selective_prediction(np.array([0.1, np.inf, 0.3, 0.2]), _ZERO_ONE_LOSSES)
    for key, value in expected.items():
        assert result[key] == pytest.approx(value)
    # The rule rejects the NaN criterion although an infinite threshold accepts +inf.
    assert result["selected:coverage"] == pytest.approx(0.75)
    assert result["selected:risk"] == pytest.approx(1 / 3)


def test_evaluate_risk_is_nan_if_nothing_is_accepted() -> None:
    result = selective_predictor(representer(_model()), selector=make_selector(threshold=-np.inf)).evaluate(
        _LABELS, None
    )

    assert result["selected:coverage"] == 0.0
    assert np.isnan(result["selected:risk"])


def test_evaluate_needs_losses_before_the_forward_pass() -> None:
    model = _CountingModel(create_categorical_distribution(PROBABILITIES))

    with pytest.raises(TypeError, match="evaluate needs losses"):
        selective_predictor(representer(model), selector=make_selector(threshold=0.5), loss=None).evaluate(
            _LABELS, None
        )
    assert model.calls == 0


def test_evaluate_rejects_losses_of_another_length() -> None:
    sp = selective_predictor(representer(_model()), selector=make_selector(threshold=0.5), loss=lambda _d, y: y)

    with pytest.raises(ValueError, match=r"same shape, got \(4,\) and \(3,\)"):
        sp.evaluate(np.zeros(3), None)


# Batching recipe.


def test_concatenated_representations_on_an_identity_model_match_one_batch() -> None:
    representation = create_categorical_distribution(PROBABILITIES)
    concatenated = np.concatenate([representation[:2], representation[2:]], axis=0)
    identity = cast(lambda r: r, predictor_type=RepresentationPredictor)

    sp = selective_predictor(representer(identity), selector=make_selector(risk=0.9, delta=0.5)).calibrate(
        np.array([0, 1, 0, 0]), concatenated
    )
    reference = selective_predictor(representer(_model()), selector=make_selector(risk=0.9, delta=0.5)).calibrate(
        np.array([0, 1, 0, 0]), None
    )

    assert type(concatenated) is type(representation)
    assert sp.selector.threshold == reference.selector.threshold


def test_subclass_overriding_only_select_is_a_selector() -> None:
    selector = _LeastUncertainSelector(k=1)

    np.testing.assert_array_equal(selector.select(MAX_PROB_COMPLEMENTS), [False, False, False, True])
    assert not selector.takes_losses


def test_selective_predictor_is_a_protocol_instance_and_predicts_through_dispatch() -> None:
    sp = selective_predictor(representer(_model()), selector=make_selector(threshold=0.3))

    assert isinstance(sp, SelectivePredictor)
    result = predict(sp, None)
    assert isinstance(result, SelectivePrediction)
    np.testing.assert_array_equal(result.accepted, sp.predict(None).accepted)


def test_calibrate_dispatch_takes_labels_first() -> None:
    sp = selective_predictor(representer(_model()), selector=make_selector(coverage=0.5))

    assert calibrate(sp, None, None) is sp
    assert sp.selector.is_calibrated


def test_representer_of_a_cast_model_is_accepted() -> None:
    model = cast(_Plain(), predictor_type="categorical_distribution_predictor")
    sp = selective_predictor(representer(model))

    assert isinstance(sp, SelectivePredictor)


def test_object_registered_to_the_protocol_gets_the_pipeline() -> None:
    class Registered:
        def __init__(self) -> None:
            inner = selective_predictor(representer(_model()), selector=make_selector(threshold=0.3))
            self.selector = inner.selector
            self.representer = inner.representer
            self.notion = inner.notion
            self.loss = inner.loss
            self.decider = inner.decider

    SelectivePredictor.register(Registered)

    result = predict(Registered(), None)

    np.testing.assert_array_equal(result.accepted, MAX_PROB_COMPLEMENTS <= 0.3)
