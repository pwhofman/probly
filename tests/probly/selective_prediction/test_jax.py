"""JAX tests for selective predictors."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from probly.quantification import LogLoss, SecondOrderZeroOneDecomposition, quantify  # noqa: E402
from probly.representation.distribution import create_categorical_distribution  # noqa: E402
from probly.representation.distribution.jax_categorical import (  # noqa: E402
    JaxCategoricalDistributionSample,
    JaxProbabilityCategoricalDistribution,
)
from probly.representation.distribution.jax_gaussian import (  # noqa: E402
    JaxGaussianDistribution,
    JaxGaussianDistributionSample,
)
from probly.selective_prediction import CoverageSelector, SelectivePredictor, ThresholdSelector  # noqa: E402

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
    result = SelectivePredictor(_JaxCategoricalModel(), ThresholdSelector(0.5)).predict(None)

    assert isinstance(result.uncertainty, jax.Array)
    assert isinstance(result.accepted, jax.Array)
    assert result.accepted.dtype == jnp.bool_
    np.testing.assert_allclose(np.asarray(result.uncertainty), 1.0 - PROBABILITIES.max(axis=-1), rtol=1e-6)
    assert result.coverage == 1.0


def test_threshold_below_criterion_rejects_on_jax() -> None:
    result = SelectivePredictor(_JaxCategoricalModel(), ThresholdSelector(0.3)).predict(None)

    np.testing.assert_array_equal(np.asarray(result.accepted), [True, False, False, True])


class _JaxSampleModel:
    """Stub ensemble that returns a fixed sample of JAX categorical distributions."""

    def predict_representation(self, _x: object) -> Any:  # noqa: ANN401
        probabilities = jnp.array([[[0.9, 0.1], [0.7, 0.3]], [[0.2, 0.8], [0.7, 0.3]]])
        return JaxCategoricalDistributionSample(
            array=JaxProbabilityCategoricalDistribution(probabilities),
            sample_axis=0,
        )


@pytest.mark.parametrize("notion", ["total", "aleatoric", "epistemic"])
def test_sample_criterion_is_zero_one_decomposition(notion: str) -> None:
    model = _JaxSampleModel()
    representation = model.predict_representation(None)

    result = SelectivePredictor(model, ThresholdSelector(0.5), notion=notion).predict(None)

    assert isinstance(result.uncertainty, jax.Array)
    np.testing.assert_allclose(
        np.asarray(result.uncertainty),
        np.asarray(SecondOrderZeroOneDecomposition(representation)[notion]),
        rtol=1e-6,
    )


@pytest.mark.parametrize("notion", ["total", "aleatoric", "epistemic"])
def test_sample_log_loss_matches_quantify(notion: str) -> None:
    model = _JaxSampleModel()
    representation = model.predict_representation(None)

    result = SelectivePredictor(model, ThresholdSelector(0.5), notion=notion, loss=LogLoss()).predict(None)

    np.testing.assert_allclose(
        np.asarray(result.uncertainty), np.asarray(quantify(representation)[notion]), rtol=1e-6, atol=1e-7
    )


def test_selective_prediction_matches_numpy_reference() -> None:
    jax_result = SelectivePredictor(_JaxCategoricalModel(), ThresholdSelector(0.5)).predict(None)
    numpy_result = SelectivePredictor(_NumpyCategoricalModel(), ThresholdSelector(0.5)).predict(None)

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

    result = SelectivePredictor(model, ThresholdSelector(1.0), notion=notion, decider=lambda rep: rep).predict(None)

    assert isinstance(result.uncertainty, jax.Array)
    np.testing.assert_allclose(np.asarray(result.uncertainty), np.asarray(quantify(representation)[notion]), rtol=1e-6)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize(("n", "coverage"), [(19, 0.9), (50, 0.3), (99, 0.5)])
def test_coverage_selector_calibrates_jax_uncertainty(n: int, coverage: float, dtype: str) -> None:
    kappa = jax.random.uniform(jax.random.key(n), (n,)).astype(dtype)
    selector = CoverageSelector(coverage).calibrate(kappa)
    expected = CoverageSelector(coverage).calibrate(np.asarray(kappa, dtype=np.float64))
    assert selector.threshold == expected.threshold
    accepted = selector.select(kappa)
    assert accepted.dtype == jnp.bool_
    assert int(accepted.sum()) >= int(np.ceil((n + 1) * coverage))


def test_coverage_selector_pipeline_calibrates_jax_model() -> None:
    sp = SelectivePredictor(_JaxCategoricalModel(), CoverageSelector(0.5))
    assert sp.calibrate(None) is sp
    reference = SelectivePredictor(_NumpyCategoricalModel(), CoverageSelector(0.5)).calibrate(None)
    assert sp.selector.threshold == pytest.approx(reference.selector.threshold)
