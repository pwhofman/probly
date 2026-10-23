"""JAX tests for selective predictors."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from ndlike.jax_functions import jax_concatenate  # noqa: E402

from probly.method import cast  # noqa: E402
from probly.predictor import RepresentationPredictor  # noqa: E402
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
from probly.representer import representer  # noqa: E402
from probly.selective_prediction import selective_predictor, selector as make_selector  # noqa: E402

PROBABILITIES = np.array([[0.9, 0.1], [0.5, 0.5], [0.6, 0.4], [0.99, 0.01]])


class _JaxCategoricalModel:
    """Uncertainty-aware stub model that returns a fixed JAX categorical distribution."""

    def predict_representation(self, _x: object) -> Any:  # noqa: ANN401
        return create_categorical_distribution(jnp.asarray(PROBABILITIES))


class _NumpyCategoricalModel:
    """NumPy reference of the JAX stub model."""

    def predict_representation(self, _x: object) -> Any:  # noqa: ANN401
        return create_categorical_distribution(PROBABILITIES)


def test_select_runs_under_jit() -> None:
    rule = make_selector(coverage=0.5).calibrate(jnp.arange(9.0))

    accepted = jax.jit(lambda kappa: rule.select(kappa).mean())(jnp.asarray([1.0, 2.0, 6.0, 7.0]))

    assert float(accepted) == pytest.approx(0.5)


def test_selective_prediction_is_jax_native() -> None:
    result = selective_predictor(representer(_JaxCategoricalModel()), selector=make_selector(threshold=0.5)).predict(
        None
    )

    assert isinstance(result.uncertainty, jax.Array)
    assert isinstance(result.accepted, jax.Array)
    assert result.accepted.dtype == jnp.bool_
    np.testing.assert_allclose(np.asarray(result.uncertainty), 1.0 - PROBABILITIES.max(axis=-1), rtol=1e-6)
    assert result.coverage == 1.0


def test_threshold_below_criterion_rejects_on_jax() -> None:
    result = selective_predictor(representer(_JaxCategoricalModel()), selector=make_selector(threshold=0.3)).predict(
        None
    )

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

    result = selective_predictor(representer(model), selector=make_selector(threshold=0.5), notion=notion).predict(None)

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

    result = selective_predictor(
        representer(model), selector=make_selector(threshold=0.5), notion=notion, loss=LogLoss()
    ).predict(None)

    np.testing.assert_allclose(
        np.asarray(result.uncertainty), np.asarray(quantify(representation)[notion]), rtol=1e-6, atol=1e-7
    )


def test_selective_prediction_matches_numpy_reference() -> None:
    jax_result = selective_predictor(
        representer(_JaxCategoricalModel()), selector=make_selector(threshold=0.5)
    ).predict(None)
    numpy_result = selective_predictor(
        representer(_NumpyCategoricalModel()), selector=make_selector(threshold=0.5)
    ).predict(None)

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

    result = selective_predictor(
        representer(model), selector=make_selector(threshold=1.0), notion=notion, decider=lambda rep: rep
    ).predict(None)

    assert isinstance(result.uncertainty, jax.Array)
    np.testing.assert_allclose(np.asarray(result.uncertainty), np.asarray(quantify(representation)[notion]), rtol=1e-6)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize(("n", "coverage"), [(19, 0.9), (50, 0.3), (99, 0.5)])
def test_coverage_selector_calibrates_jax_uncertainty(n: int, coverage: float, dtype: str) -> None:
    kappa = jax.random.uniform(jax.random.key(n), (n,)).astype(dtype)
    selector = make_selector(coverage=coverage).calibrate(kappa)
    expected = make_selector(coverage=coverage).calibrate(np.asarray(kappa, dtype=np.float64))
    assert selector.threshold == expected.threshold
    accepted = selector.select(kappa)
    assert accepted.dtype == jnp.bool_
    assert int(accepted.sum()) >= int(np.ceil((n + 1) * coverage))


def test_coverage_selector_pipeline_calibrates_jax_model() -> None:
    sp = selective_predictor(representer(_JaxCategoricalModel()), selector=make_selector(coverage=0.5))
    assert sp.calibrate(None, None) is sp
    reference = selective_predictor(
        representer(_NumpyCategoricalModel()), selector=make_selector(coverage=0.5)
    ).calibrate(None, None)
    assert sp.selector.threshold == pytest.approx(reference.selector.threshold)


def test_sgr_selector_calibrates_jax_arrays() -> None:
    kappa = jax.random.uniform(jax.random.key(0), (400,))
    losses = (jax.random.uniform(jax.random.key(1), (400,)) < 0.2 * kappa**2).astype(jnp.float32)
    selector = make_selector(risk=0.1, delta=0.2).calibrate(kappa, losses)
    expected = make_selector(risk=0.1, delta=0.2).calibrate(
        np.asarray(kappa, dtype=np.float64), np.asarray(losses, dtype=np.float64)
    )
    assert selector.threshold == expected.threshold
    assert selector.select(kappa).dtype == jnp.bool_


def test_sgr_selector_pipeline_calibrates_jax_model_with_targets() -> None:
    targets = np.array([0, 1, 0, 0])
    sp = selective_predictor(representer(_JaxCategoricalModel()), selector=make_selector(risk=0.9, delta=0.5))
    assert sp.calibrate(jnp.asarray(targets), None) is sp
    reference = selective_predictor(
        representer(_NumpyCategoricalModel()), selector=make_selector(risk=0.9, delta=0.5)
    ).calibrate(targets, None)
    assert sp.selector.threshold == pytest.approx(reference.selector.threshold)


def test_evaluate_matches_numpy_reference() -> None:
    labels = np.array([0, 1, 0, 1])
    jax_result = selective_predictor(
        representer(_JaxCategoricalModel()), selector=make_selector(threshold=0.3)
    ).evaluate(jnp.asarray(labels), None)
    numpy_result = selective_predictor(
        representer(_NumpyCategoricalModel()), selector=make_selector(threshold=0.3)
    ).evaluate(labels, None)

    assert jax_result.keys() == numpy_result.keys()
    assert jax_result == pytest.approx(numpy_result, rel=1e-6)


def test_jax_concatenated_representations_on_an_identity_model_match_one_batch() -> None:
    representation = _JaxSampleModel().predict_representation(None)
    concatenated = jax_concatenate([representation[:, :1], representation[:, 1:]], axis=1)
    identity = cast(lambda r: r, predictor_type=RepresentationPredictor)

    result = selective_predictor(
        representer(identity), selector=make_selector(threshold=0.5), notion="epistemic"
    ).predict(concatenated)
    expected = selective_predictor(
        representer(_JaxSampleModel()), selector=make_selector(threshold=0.5), notion="epistemic"
    ).predict(None)

    assert type(concatenated) is type(representation)
    np.testing.assert_allclose(np.asarray(result.uncertainty), np.asarray(expected.uncertainty), rtol=1e-6)
