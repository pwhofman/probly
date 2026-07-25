"""Tests for Mahalanobis uncertainty quantification with sklearn."""

from __future__ import annotations

import numpy as np
import pytest

from probly.method.mahalanobis import mahalanobis
from probly.predictor import predict
from probly.quantification import decompose
from probly.quantification.notion import AleatoricUncertainty, EpistemicUncertainty

pytest.importorskip("sklearn")

from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier

from probly.method.mahalanobis.sklearn import SklearnMahalanobisPredictor

NUM_SAMPLES = 300
HIDDEN_SIZES = (16, 8)


@pytest.fixture
def fitted_model() -> SklearnMahalanobisPredictor:
    """A Mahalanobis predictor with fitted heads over a three-class two-moons batch."""
    x, y = make_moons(n_samples=NUM_SAMPLES, noise=0.1, random_state=0)
    labels = (x[:, 0] > 0).astype(int) + y
    base = MLPClassifier(hidden_layer_sizes=HIDDEN_SIZES, max_iter=800, random_state=0).fit(x, labels)
    model = mahalanobis(base)
    model.fit_mahalanobis_heads(x, labels)
    return model


@pytest.fixture
def in_distribution() -> np.ndarray:
    """Inputs drawn from the training distribution."""
    return make_moons(n_samples=50, noise=0.1, random_state=1)[0]


def test_decomposition_exposes_epistemic_only(
    fitted_model: SklearnMahalanobisPredictor, in_distribution: np.ndarray
) -> None:
    """The Mahalanobis decomposition exposes only the epistemic OOD score (no aleatoric slot)."""
    decomposition = decompose(predict(fitted_model, in_distribution))
    assert decomposition[EpistemicUncertainty].shape == (len(in_distribution),)
    with pytest.raises(KeyError):
        _ = decomposition[AleatoricUncertainty]


def test_epistemic_higher_for_far_inputs(
    fitted_model: SklearnMahalanobisPredictor, in_distribution: np.ndarray
) -> None:
    """Inputs far from the fitted class centroids get a higher epistemic score."""
    far = np.random.default_rng(1).normal(size=(50, 2)) * 5 + 20

    in_eu = decompose(predict(fitted_model, in_distribution)).epistemic
    out_eu = decompose(predict(fitted_model, far)).epistemic
    assert out_eu.mean() > in_eu.mean()


def test_uniform_weights_before_calibration(
    fitted_model: SklearnMahalanobisPredictor, in_distribution: np.ndarray
) -> None:
    """Before calibration the combiner uses negative-unit weights (works without fit_combiner)."""
    np.testing.assert_allclose(fitted_model.combiner_weight, -np.ones_like(fitted_model.combiner_weight))
    epistemic = decompose(predict(fitted_model, in_distribution)).epistemic
    assert np.isfinite(epistemic).all()
