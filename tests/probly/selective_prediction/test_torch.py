"""PyTorch tests for selective predictors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from probly import method  # noqa: E402
from probly.decider import categorical_from_maximin  # noqa: E402
from probly.method import cast  # noqa: E402
from probly.metrics import log_loss_per_instance  # noqa: E402
from probly.predictor import RepresentationPredictor  # noqa: E402
from probly.quantification import SecondOrderZeroOneDecomposition, quantify  # noqa: E402
from probly.representation.distribution import (  # noqa: E402
    CategoricalDistribution,
    CategoricalDistributionSample,
    DirichletDistribution,
    create_categorical_distribution,
    create_dirichlet_distribution_from_alphas,
)
from probly.representer import representer  # noqa: E402
from probly.representer.sampler import Sampler  # noqa: E402
from probly.selective_prediction import selective_predictor, selector as make_selector  # noqa: E402
from probly.transformation import dropout, ensemble  # noqa: E402
from probly.transformation.ensemble import EnsemblePredictor  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable


def _ensemble_model() -> EnsemblePredictor:
    torch.manual_seed(0)
    return ensemble(nn.Linear(4, 3), num_members=3, predictor_type="logit_classifier")


def test_ensemble_selective_prediction_is_torch_native() -> None:
    model = _ensemble_model()
    x = torch.randn(8, 4)

    with torch.no_grad():
        result = selective_predictor(representer(model), selector=make_selector(threshold=0.6)).predict(x)
        expected_uncertainty = 1.0 - representer(model).represent(x).sample_mean().probabilities.max(dim=-1).values

    assert isinstance(result.uncertainty, torch.Tensor)
    assert isinstance(result.accepted, torch.Tensor)
    assert result.accepted.dtype == torch.bool
    assert result.accepted.shape == (8,)
    torch.testing.assert_close(result.uncertainty, expected_uncertainty)
    torch.testing.assert_close(result.accepted, expected_uncertainty <= 0.6)


@pytest.mark.parametrize("notion", ["total", "aleatoric", "epistemic"])
def test_ensemble_log_loss_matches_quantify(notion: str) -> None:
    model = _ensemble_model()
    x = torch.randn(8, 4)

    with torch.no_grad():
        result = selective_predictor(
            representer(model), selector=make_selector(threshold=0.6, notion=notion), loss=log_loss_per_instance
        ).predict(x)
        expected = quantify(representer(model).represent(x))[notion]

    torch.testing.assert_close(result.uncertainty, expected)


class _RepresentationModel:
    """Uncertainty-aware stub model that returns a fixed representation."""

    def __init__(self, representation: object) -> None:
        self.representation = representation

    def predict_representation(self, _x: object) -> object:
        return self.representation


def test_single_distribution_criterion_is_torch_native() -> None:
    probabilities = torch.tensor([[0.9, 0.1], [0.5, 0.5], [0.6, 0.4], [0.99, 0.01]])
    model = _RepresentationModel(create_categorical_distribution(probabilities))

    result = selective_predictor(representer(model), selector=make_selector(threshold=0.3)).predict(None)

    assert isinstance(result.uncertainty, torch.Tensor)
    torch.testing.assert_close(result.uncertainty, 1.0 - probabilities.max(dim=-1).values)
    torch.testing.assert_close(result.accepted, torch.tensor([True, False, False, True]))


def test_dirichlet_criterion_is_zero_one_total() -> None:
    dirichlet = create_dirichlet_distribution_from_alphas(torch.tensor([[8.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))

    result = selective_predictor(
        representer(_RepresentationModel(dirichlet)), selector=make_selector(threshold=0.5)
    ).predict(None)

    assert isinstance(result.uncertainty, torch.Tensor)
    torch.testing.assert_close(result.uncertainty, SecondOrderZeroOneDecomposition(dirichlet).total)
    torch.testing.assert_close(result.accepted, torch.tensor([True, False]))


def test_ensemble_selective_prediction_matches_numpy_threshold() -> None:
    model = _ensemble_model()
    x = torch.randn(8, 4)
    with torch.no_grad():
        uncertainty = (
            selective_predictor(representer(model), selector=make_selector(threshold=np.inf)).predict(x).uncertainty
        )
        tau = float(uncertainty.median())
        result = selective_predictor(representer(model), selector=make_selector(threshold=tau)).predict(x)

    np.testing.assert_array_equal(result.accepted.numpy(), uncertainty.numpy() <= tau)
    assert result.coverage == float((uncertainty <= tau).float().mean())


def test_dropout_model_is_sampled_with_the_configured_representer() -> None:
    torch.manual_seed(0)
    model = dropout(
        nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3)), p=0.5, predictor_type="logit_classifier"
    )
    x = torch.randn(8, 4)

    sp = selective_predictor(
        representer(model, num_samples=7), selector=make_selector(threshold=0.6, notion="epistemic")
    )
    with torch.no_grad():
        result = sp.predict(x)

    assert isinstance(sp.representer, Sampler)
    assert sp.representer.num_samples == 7
    assert result.accepted.dtype == torch.bool
    assert result.accepted.shape == (8,)
    assert bool((result.uncertainty > 0).all())


class _GaussianRegressor(nn.Module):
    """Network returning the mean and variance of a Gaussian predictive distribution."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.linear(x)
        return out[:, 0], nn.functional.softplus(out[:, 1])


def test_regression_ensemble_selects_on_epistemic_uncertainty() -> None:
    torch.manual_seed(0)
    model = ensemble(_GaussianRegressor(), num_members=3, predictor_type="gaussian_distribution_predictor")
    x = torch.randn(8, 4)

    with torch.no_grad():
        result = selective_predictor(
            representer(model),
            selector=make_selector(threshold=0.5, notion=lambda rep: quantify(rep).epistemic),
            decider=lambda rep: rep,
        ).predict(x)
        representation = representer(model).represent(x)

    torch.testing.assert_close(result.uncertainty, quantify(representation).epistemic)
    torch.testing.assert_close(result.accepted, result.uncertainty <= 0.5)


def _classifier_base() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))


_ALL_NOTIONS: dict[str, type[Exception] | None] = {"total": None, "aleatoric": None, "epistemic": None}
_TOTAL_ONLY: dict[str, type[Exception] | None] = {"total": None, "aleatoric": KeyError, "epistemic": KeyError}
_UNSUPPORTED: dict[str, type[Exception] | None] = dict.fromkeys(_ALL_NOTIONS, NotImplementedError)

# Method name: (transformation, representer kwargs, branch of the default loss, error per notion or None).
_METHOD_MATRIX: dict[str, tuple[Callable[[nn.Module], Any], dict[str, Any], str, dict[str, type[Exception] | None]]] = {
    "ensemble": (
        lambda m: method.ensemble(m, num_members=3, predictor_type="logit_classifier"),
        {},
        "sample",
        _ALL_NOTIONS,
    ),
    "dropout": (
        lambda m: method.dropout(m, p=0.2, predictor_type="logit_classifier"),
        {"num_samples": 5},
        "sample",
        _ALL_NOTIONS,
    ),
    "dropconnect": (
        lambda m: method.dropconnect(m, predictor_type="logit_classifier"),
        {"num_samples": 5},
        "sample",
        _ALL_NOTIONS,
    ),
    "bayesian": (
        lambda m: method.bayesian(m, predictor_type="logit_classifier"),
        {"num_samples": 5},
        "sample",
        _ALL_NOTIONS,
    ),
    "batchensemble": (
        lambda m: method.batchensemble(m, num_members=3, predictor_type="logit_classifier"),
        {},
        "sample",
        _ALL_NOTIONS,
    ),
    "masksembles": (lambda m: method.masksembles(m, predictor_type="logit_classifier"), {}, "sample", _ALL_NOTIONS),
    "subensemble": (
        lambda m: method.subensemble(m, num_heads=3, predictor_type="logit_classifier"),
        {},
        "sample",
        _ALL_NOTIONS,
    ),
    "sngp": (
        lambda m: method.sngp(m, num_random_features=32, predictor_type="logit_classifier"),
        {},
        "sample",
        _ALL_NOTIONS,
    ),
    "dare": (lambda m: method.dare(m, num_members=3, predictor_type="logit_classifier"), {}, "sample", _ALL_NOTIONS),
    "vbll": (method.vbll, {}, "sample", _ALL_NOTIONS),
    "cast": (lambda m: method.cast(m, predictor_type="logit_classifier"), {}, "categorical", _TOTAL_ONLY),
    "het_net": (
        lambda m: method.het_net(m, num_samples=5, predictor_type="logit_distribution_predictor"),
        {},
        "categorical",
        _TOTAL_ONLY,
    ),
    "g_vbll": (method.g_vbll, {}, "categorical", _TOTAL_ONLY),
    # SecondOrderZeroOneDecomposition has no epistemic part for Dirichlet distributions yet.
    "evidential_classification": (
        method.evidential_classification,
        {},
        "dirichlet",
        {"total": None, "aleatoric": None, "epistemic": NotImplementedError},
    ),
    "credal_ensembling": (
        lambda m: method.credal_ensembling(m, num_members=3, predictor_type="logit_classifier"),
        {},
        "unsupported",
        _UNSUPPORTED,
    ),
    "credal_wrapper": (
        lambda m: method.credal_wrapper(m, num_members=3, predictor_type="logit_classifier"),
        {},
        "unsupported",
        _UNSUPPORTED,
    ),
    "credal_relative_likelihood": (
        lambda m: method.credal_relative_likelihood(m, num_members=3, predictor_type="logit_classifier"),
        {},
        "unsupported",
        _UNSUPPORTED,
    ),
    "credal_dro": (
        lambda m: method.credal_dro(m, num_members=3, predictor_type="logit_classifier"),
        {},
        "unsupported",
        _UNSUPPORTED,
    ),
    "credal_net": (lambda m: method.credal_net(m, predictor_type="logit_classifier"), {}, "unsupported", _UNSUPPORTED),
    "credal_bnn": (
        lambda m: method.credal_bnn(m, num_members=3, predictor_type="logit_classifier"),
        {},
        "unsupported",
        _UNSUPPORTED,
    ),
    "duq": (lambda m: method.duq(m, centroid_size=4), {}, "unsupported", _UNSUPPORTED),
    "ddu": (method.ddu, {}, "unsupported", _UNSUPPORTED),
}


def _branch(representation: object) -> str:
    if isinstance(representation, CategoricalDistributionSample):
        return "sample"
    if isinstance(representation, CategoricalDistribution):
        return "categorical"
    if isinstance(representation, DirichletDistribution):
        return "dirichlet"
    return "unsupported"


@pytest.mark.filterwarnings("ignore:No residual connections detected:UserWarning")
@pytest.mark.parametrize("notion", ["total", "aleatoric", "epistemic"])
@pytest.mark.parametrize("name", list(_METHOD_MATRIX))
def test_default_predictor_on_method(name: str, notion: str) -> None:
    transformation, representer_kwargs, branch, errors = _METHOD_MATRIX[name]
    torch.manual_seed(0)
    model = transformation(_classifier_base())
    x = torch.randn(8, 4)
    predictor = selective_predictor(
        representer(model, **representer_kwargs), selector=make_selector(threshold=0.5, notion=notion)
    )

    with torch.no_grad():
        torch.manual_seed(1)
        representation = predictor.representer.represent(x)
        assert _branch(representation) == branch
        error = errors[notion]
        if error is not None:
            with pytest.raises(error):
                predictor.predict(x)
            return
        torch.manual_seed(1)
        result = predictor.predict(x)

    assert result.uncertainty.shape == (8,)
    assert result.accepted.dtype == torch.bool
    torch.testing.assert_close(result.accepted, result.uncertainty <= 0.5)
    if notion == "total":
        # Chow's criterion: one minus the maximum probability of the decision.
        torch.testing.assert_close(result.uncertainty, 1.0 - result.decision.probabilities.max(dim=-1).values)


def test_non_default_decider_on_credal_model() -> None:
    torch.manual_seed(0)
    model = method.credal_bnn(_classifier_base(), num_members=3, predictor_type="logit_classifier")
    x = torch.randn(8, 4)

    with torch.no_grad():
        torch.manual_seed(1)
        result = selective_predictor(
            representer(model),
            selector=make_selector(threshold=1.0, notion=lambda rep: quantify(rep).total),
            decider=categorical_from_maximin,
        ).predict(x)
        torch.manual_seed(1)
        representation = representer(model).represent(x)

    assert isinstance(result.decision, CategoricalDistribution)
    torch.testing.assert_close(result.decision.probabilities, categorical_from_maximin(representation).probabilities)
    torch.testing.assert_close(result.uncertainty, quantify(representation).total)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(("n", "coverage"), [(19, 0.9), (50, 0.3), (99, 0.5)])
def test_coverage_selector_calibrates_torch_uncertainty(n: int, coverage: float, dtype: torch.dtype) -> None:
    kappa = torch.rand(n, generator=torch.Generator().manual_seed(n)).to(dtype).requires_grad_()
    selector = make_selector(coverage=coverage).calibrate(kappa)
    expected = make_selector(coverage=coverage).calibrate(kappa.detach().double().numpy())
    assert selector.threshold == expected.threshold
    accepted = selector.select(kappa.detach())
    assert accepted.dtype == torch.bool
    assert accepted.sum() >= int(np.ceil((n + 1) * coverage))


def test_coverage_selector_pipeline_calibrates_ensemble() -> None:
    model = _ensemble_model()
    x_cal, x_test = torch.randn(200, 4), torch.randn(100, 4)
    sp = selective_predictor(representer(model), selector=make_selector(coverage=0.8))
    with torch.no_grad():
        assert sp.calibrate(None, x_cal) is sp
        result = sp.predict(x_test)
        kappa_cal = (
            selective_predictor(representer(model), selector=make_selector(threshold=1.0)).predict(x_cal).uncertainty
        )
    expected = make_selector(coverage=0.8).calibrate(kappa_cal).threshold
    assert sp.selector.threshold == expected
    assert isinstance(result.accepted, torch.Tensor)
    assert torch.equal(result.accepted, result.uncertainty <= expected)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_sgr_selector_calibrates_torch_uncertainty_and_losses(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(0)
    kappa = torch.rand(400, generator=generator, dtype=torch.float64).to(dtype)
    losses = (torch.rand(400, generator=generator) < 0.2 * kappa.double() ** 2).to(dtype)
    kappa.requires_grad_()
    selector = make_selector(risk=0.1, delta=0.2).calibrate(kappa, losses)
    expected = make_selector(risk=0.1, delta=0.2).calibrate(kappa.detach().double().numpy(), losses.double().numpy())
    assert selector.threshold == expected.threshold
    np.testing.assert_equal(selector.bound, expected.bound)
    assert selector.select(kappa.detach()).dtype == torch.bool


def test_sgr_selector_pipeline_calibrates_ensemble_with_targets() -> None:
    model = _ensemble_model()
    x_cal = torch.randn(300, 4)
    with torch.no_grad():
        decision = selective_predictor(representer(model), selector=make_selector(threshold=1.0)).predict(x_cal)
        # Labels that agree with the decision for the more certain instances.
        targets = decision.decision.probabilities.argmax(dim=-1)
        flip = decision.uncertainty > decision.uncertainty.median()
        targets = torch.where(flip, (targets + 1) % 3, targets)
        sp = selective_predictor(representer(model), selector=make_selector(risk=0.2, delta=0.2))
        assert sp.calibrate(targets, x_cal) is sp
    losses = (decision.decision.probabilities.argmax(dim=-1) != targets).double().numpy()
    expected = make_selector(risk=0.2, delta=0.2).calibrate(decision.uncertainty.double().numpy(), losses)
    assert sp.selector.threshold == expected.threshold
    assert sp.selector.threshold is not None


def test_evaluate_and_predict_agree_on_a_float32_criterion_at_a_fixed_threshold() -> None:
    kappa = torch.tensor([0.3, 0.3, 0.9, 0.9], dtype=torch.float32)
    representation = create_categorical_distribution(torch.full((4, 2), 0.5))
    sp = selective_predictor(
        representer(cast(lambda r: r, predictor_type=RepresentationPredictor)),
        selector=make_selector(threshold=0.3, notion=lambda _rep: kappa),
        loss=lambda _d, y: y,
    )
    accepted = sp.predict(representation).accepted
    result = sp.evaluate(torch.zeros(4), representation)
    assert float(accepted.double().mean()) == result["selected:coverage"] == 0.5


def test_ensemble_evaluate_matches_predict() -> None:
    model = _ensemble_model()
    x = torch.randn(50, 4)
    sp = selective_predictor(representer(model), selector=make_selector(threshold=0.5))
    with torch.no_grad():
        result = sp.predict(x)
        targets = result.decision.probabilities.argmax(dim=-1)
        targets[:10] = (targets[:10] + 1) % 3
        metrics = sp.evaluate(targets, x)

    errors = (result.decision.probabilities.argmax(dim=-1) != targets).double()
    assert metrics["risk@1"] == pytest.approx(0.2)
    assert metrics["selected:coverage"] == pytest.approx(result.coverage)
    assert metrics["selected:risk"] == pytest.approx(float(errors[result.accepted].mean()))


def test_regression_recipe_on_a_dropout_model() -> None:
    torch.manual_seed(0)
    model = dropout(nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1)), p=0.2)
    x_cal, x_test, y_test = torch.randn(50, 4), torch.randn(30, 4), torch.randn(30)
    sp = selective_predictor(
        representer(model, num_samples=5),
        selector=make_selector(coverage=0.8, notion=lambda s: s.sample_var().squeeze(-1)),
        decider=lambda s: s.sample_mean().squeeze(-1),
        loss=lambda d, y: (d - y) ** 2,
    )
    with torch.no_grad():
        sp.calibrate(None, x_cal)
        result = sp.predict(x_test)
        metrics = sp.evaluate(y_test, x_test)

    assert sp.selector.threshold is not None
    assert result.uncertainty.shape == (30,)
    assert result.decision.shape == (30,)
    assert set(metrics) >= {"aurc", "risk@1", "selected:coverage", "selected:risk"}
    assert "accuracy@1" not in metrics


def test_concatenated_torch_representations_on_an_identity_model_match_one_batch() -> None:
    model = _ensemble_model()
    x = torch.randn(20, 4)
    targets = torch.randint(0, 3, (20,))
    with torch.no_grad():
        rep = representer(model)
        concatenated = torch.cat([rep.represent(x[:10]), rep.represent(x[10:])], dim=0)
        identity = cast(lambda r: r, predictor_type=RepresentationPredictor)
        result = selective_predictor(representer(identity), selector=make_selector(threshold=0.6)).evaluate(
            targets, concatenated
        )
        expected = selective_predictor(representer(model), selector=make_selector(threshold=0.6)).evaluate(targets, x)

    assert isinstance(concatenated, CategoricalDistributionSample)
    assert 0.0 < result["selected:coverage"] < 1.0
    assert result == pytest.approx(expected)


def test_error_note_names_the_torch_representation() -> None:
    def notion(_representation: object) -> torch.Tensor:
        msg = "user error"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="user error") as info, torch.no_grad():
        selective_predictor(
            representer(_ensemble_model()), selector=make_selector(threshold=0.5, notion=notion)
        ).predict(torch.randn(8, 4))
    assert "notion receives the representation (TorchSample), not the model output" in info.value.__notes__[0]
