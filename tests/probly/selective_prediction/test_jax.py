"""JAX tests for selective predictors."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from probly.quantification import quantify  # noqa: E402
from probly.representation.distribution import create_categorical_distribution  # noqa: E402
from probly.representation.distribution.jax_gaussian import (  # noqa: E402
    JaxGaussianDistribution,
    JaxGaussianDistributionSample,
)
from probly.selective_prediction import ThresholdSelectivePredictor  # noqa: E402

PROBABILITIES = np.array([[0.9, 0.1], [0.5, 0.5], [0.6, 0.4], [0.99, 0.01]])


class _JaxCategoricalModel:
    """Uncertainty-aware stub model that returns a fixed JAX categorical distribution."""

    def predict_representation(self, _x: object) -> Any:  # noqa: ANN401
        return create_categorical_distribution(jnp.asarray(PROBABILITIES))


class _NumpyCategoricalModel:
    """NumPy reference of the JAX stub model."""

    def predict_representation(self, _x: object) -> Any:  # noqa: ANN401
        return create_categorical_distribution(PROBABILITIES)


def test_selective_prediction_is_jax_native() -> None:
    result = ThresholdSelectivePredictor(_JaxCategoricalModel(), threshold=0.5).predict(None)

    assert isinstance(result.uncertainty, jax.Array)
    assert isinstance(result.accepted, jax.Array)
    assert result.accepted.dtype == jnp.bool_
    assert result.coverage == 0.5


def test_selective_prediction_matches_numpy_reference() -> None:
    jax_result = ThresholdSelectivePredictor(_JaxCategoricalModel(), threshold=0.5).predict(None)
    numpy_result = ThresholdSelectivePredictor(_NumpyCategoricalModel(), threshold=0.5).predict(None)

    np.testing.assert_allclose(np.asarray(jax_result.uncertainty), numpy_result.uncertainty, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(jax_result.accepted), numpy_result.accepted)


class _JaxRegressionModel:
    """Stub regression ensemble that returns a fixed sample of JAX Gaussian distributions."""

    def predict_representation(self, _x: object) -> Any:  # noqa: ANN401
        means = jnp.array([[0.0, 1.0, 2.0], [2.0, 1.0, 2.5]])
        variances = jnp.array([[1.0, 0.5, 0.1], [2.0, 0.5, 0.2]])
        return JaxGaussianDistributionSample(JaxGaussianDistribution(mean=means, var=variances), sample_axis=0)


@pytest.mark.parametrize("notion", ["total", "aleatoric", "epistemic"])
def test_regression_criterion_comes_from_quantify(notion: str) -> None:
    model = _JaxRegressionModel()
    representation = model.predict_representation(None)

    result = ThresholdSelectivePredictor(model, threshold=1.0, notion=notion, decider=lambda rep: rep).predict(None)

    assert isinstance(result.uncertainty, jax.Array)
    np.testing.assert_allclose(np.asarray(result.uncertainty), np.asarray(quantify(representation)[notion]), rtol=1e-6)
