"""Tests for Mahalanobis uncertainty quantification with torch."""

from __future__ import annotations

import pytest

from probly.method.mahalanobis import mahalanobis
from probly.predictor import predict
from probly.quantification import decompose
from probly.quantification.notion import AleatoricUncertainty, EpistemicUncertainty

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from probly.method.mahalanobis.torch import TorchMahalanobisPredictor  # noqa: E402

# ``torch_custom_model``: Linear(10, 20) -> ReLU -> Linear(20, 4) -> Softmax.
IN_FEATURES = 10
NUM_CLASSES = 4


@pytest.fixture
def fitted_model(torch_custom_model: nn.Module) -> TorchMahalanobisPredictor:
    torch.manual_seed(0)
    model = mahalanobis(torch_custom_model)
    x = torch.randn(90, IN_FEATURES)
    y = torch.randint(0, NUM_CLASSES, (90,))
    model.fit_mahalanobis_heads(x, y)
    return model


def test_decomposition_exposes_epistemic_only(fitted_model: TorchMahalanobisPredictor) -> None:
    """The Mahalanobis decomposition exposes only the epistemic OOD score (no aleatoric slot)."""
    x = torch.randn(10, IN_FEATURES)
    decomposition = decompose(predict(fitted_model, x))
    assert decomposition[EpistemicUncertainty].shape == (10,)
    with pytest.raises(KeyError):
        _ = decomposition[AleatoricUncertainty]


def test_epistemic_higher_for_far_inputs(fitted_model: TorchMahalanobisPredictor) -> None:
    """Inputs far from the fitted class centroids get a higher epistemic score."""
    torch.manual_seed(1)
    in_dist = torch.randn(50, IN_FEATURES)
    out_dist = torch.randn(50, IN_FEATURES) * 5 + 20

    in_eu = decompose(predict(fitted_model, in_dist)).epistemic
    out_eu = decompose(predict(fitted_model, out_dist)).epistemic
    assert out_eu.mean() > in_eu.mean()


def test_uniform_weights_before_calibration(fitted_model: TorchMahalanobisPredictor) -> None:
    """Before calibration the combiner uses negative-unit weights (works without fit_combiner)."""
    assert torch.equal(fitted_model.combiner_weight, -torch.ones_like(fitted_model.combiner_weight))
    eu = decompose(predict(fitted_model, torch.randn(5, IN_FEATURES))).epistemic
    assert torch.isfinite(eu).all()
