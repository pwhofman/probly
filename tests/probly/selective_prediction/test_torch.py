"""PyTorch tests for selective predictors."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from probly.quantification import quantify  # noqa: E402
from probly.representer import representer  # noqa: E402
from probly.representer.sampler import Sampler  # noqa: E402
from probly.selective_prediction import ThresholdSelectivePredictor  # noqa: E402
from probly.transformation import dropout, ensemble  # noqa: E402
from probly.transformation.ensemble import EnsemblePredictor  # noqa: E402


def _ensemble_model() -> EnsemblePredictor:
    torch.manual_seed(0)
    return ensemble(nn.Linear(4, 3), num_members=3, predictor_type="logit_classifier")


def test_ensemble_selective_prediction_is_torch_native() -> None:
    model = _ensemble_model()
    x = torch.randn(8, 4)

    with torch.no_grad():
        result = ThresholdSelectivePredictor(model, threshold=0.6).predict(x)
        expected_uncertainty = quantify(representer(model).represent(x)).total

    assert isinstance(result.uncertainty, torch.Tensor)
    assert isinstance(result.accepted, torch.Tensor)
    assert result.accepted.dtype == torch.bool
    assert result.accepted.shape == (8,)
    torch.testing.assert_close(result.uncertainty, expected_uncertainty)
    torch.testing.assert_close(result.accepted, expected_uncertainty <= 0.6)


def test_ensemble_selective_prediction_matches_numpy_threshold() -> None:
    model = _ensemble_model()
    x = torch.randn(8, 4)
    with torch.no_grad():
        uncertainty = ThresholdSelectivePredictor(model, threshold=np.inf).predict(x).uncertainty
        tau = float(uncertainty.median())
        result = ThresholdSelectivePredictor(model, threshold=tau).predict(x)

    np.testing.assert_array_equal(result.accepted.numpy(), uncertainty.numpy() <= tau)
    assert result.coverage == float((uncertainty <= tau).float().mean())


def test_dropout_model_is_sampled_with_representer_kwargs() -> None:
    torch.manual_seed(0)
    model = dropout(
        nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3)), p=0.5, predictor_type="logit_classifier"
    )
    x = torch.randn(8, 4)

    sp = ThresholdSelectivePredictor(model, threshold=0.6, notion="epistemic", representer_kwargs={"num_samples": 7})
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
        result = ThresholdSelectivePredictor(model, threshold=0.5, notion="epistemic", decider=lambda rep: rep).predict(
            x
        )
        representation = representer(model).represent(x)

    torch.testing.assert_close(result.uncertainty, quantify(representation).epistemic)
    torch.testing.assert_close(result.accepted, result.uncertainty <= 0.5)
