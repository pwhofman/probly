"""PyTorch tests for selective predictors."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from probly.quantification import LogLoss, SecondOrderZeroOneDecomposition, quantify  # noqa: E402
from probly.representation.distribution import (  # noqa: E402
    create_categorical_distribution,
    create_dirichlet_distribution_from_alphas,
)
from probly.representer import representer  # noqa: E402
from probly.representer.sampler import Sampler  # noqa: E402
from probly.selective_prediction import SelectivePredictor, ThresholdSelector  # noqa: E402
from probly.transformation import dropout, ensemble  # noqa: E402
from probly.transformation.ensemble import EnsemblePredictor  # noqa: E402


def _ensemble_model() -> EnsemblePredictor:
    torch.manual_seed(0)
    return ensemble(nn.Linear(4, 3), num_members=3, predictor_type="logit_classifier")


def test_ensemble_selective_prediction_is_torch_native() -> None:
    model = _ensemble_model()
    x = torch.randn(8, 4)

    with torch.no_grad():
        result = SelectivePredictor(model, ThresholdSelector(0.6)).predict(x)
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
        result = SelectivePredictor(model, ThresholdSelector(0.6), notion=notion, loss=LogLoss()).predict(x)
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

    result = SelectivePredictor(model, ThresholdSelector(0.3)).predict(None)

    assert isinstance(result.uncertainty, torch.Tensor)
    torch.testing.assert_close(result.uncertainty, 1.0 - probabilities.max(dim=-1).values)
    torch.testing.assert_close(result.accepted, torch.tensor([True, False, False, True]))


def test_dirichlet_criterion_is_zero_one_total() -> None:
    dirichlet = create_dirichlet_distribution_from_alphas(torch.tensor([[8.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))

    result = SelectivePredictor(_RepresentationModel(dirichlet), ThresholdSelector(0.5)).predict(None)

    assert isinstance(result.uncertainty, torch.Tensor)
    torch.testing.assert_close(result.uncertainty, SecondOrderZeroOneDecomposition(dirichlet).total)
    torch.testing.assert_close(result.accepted, torch.tensor([True, False]))


def test_ensemble_selective_prediction_matches_numpy_threshold() -> None:
    model = _ensemble_model()
    x = torch.randn(8, 4)
    with torch.no_grad():
        uncertainty = SelectivePredictor(model, ThresholdSelector(np.inf)).predict(x).uncertainty
        tau = float(uncertainty.median())
        result = SelectivePredictor(model, ThresholdSelector(tau)).predict(x)

    np.testing.assert_array_equal(result.accepted.numpy(), uncertainty.numpy() <= tau)
    assert result.coverage == float((uncertainty <= tau).float().mean())


def test_dropout_model_is_sampled_with_representer_kwargs() -> None:
    torch.manual_seed(0)
    model = dropout(
        nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3)), p=0.5, predictor_type="logit_classifier"
    )
    x = torch.randn(8, 4)

    sp = SelectivePredictor(model, ThresholdSelector(0.6), notion="epistemic", representer_kwargs={"num_samples": 7})
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
        result = SelectivePredictor(model, ThresholdSelector(0.5), notion="epistemic", decider=lambda rep: rep).predict(
            x
        )
        representation = representer(model).represent(x)

    torch.testing.assert_close(result.uncertainty, quantify(representation).epistemic)
    torch.testing.assert_close(result.accepted, result.uncertainty <= 0.5)
