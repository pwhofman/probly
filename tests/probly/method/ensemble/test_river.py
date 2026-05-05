"""Tests for river ARF ensemble integration."""

from __future__ import annotations

import numpy as np
import pytest

from probly.quantification.decomposition.decomposition import TotalDecomposition
from probly.representation.sample.array import ArraySample

pytest.importorskip("river")

from river.datasets import synth
from river.forest import ARFClassifier, ARFRegressor

from probly.predictor import predict_raw
from probly.quantification import measure, quantify
from probly.representation.distribution.array_categorical import (
    ArrayCategoricalDistribution,
    ArrayCategoricalDistributionSample,
    ArrayProbabilityCategoricalDistribution,
)
from probly.representation.sample._common import create_sample
from probly.representer import representer


@pytest.fixture
def trained_arf():
    """Train a small ARF on Agrawal stream and return (model, last_x)."""
    arf = ARFClassifier(n_models=5, seed=42)
    stream = synth.Agrawal(classification_function=0, seed=42)
    last_x: dict = {}
    for x, y in stream.take(200):
        arf.learn_one(x, y)
        last_x = x
    return arf, last_x


class TestPredictRaw:
    def test_returns_list_of_distributions(self, trained_arf):
        arf, x = trained_arf
        result = predict_raw(arf, x)

        assert isinstance(result, list)
        assert len(result) == arf.n_models
        for dist in result:
            assert isinstance(dist, ArrayCategoricalDistribution)

    def test_distributions_are_valid_probabilities(self, trained_arf):
        arf, x = trained_arf
        result = predict_raw(arf, x)

        for dist in result:
            probs = dist.probabilities
            assert np.all(probs >= 0)
            np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-10)

    def test_all_distributions_same_num_classes(self, trained_arf):
        arf, x = trained_arf
        result = predict_raw(arf, x)
        num_classes = result[0].num_classes
        for dist in result:
            assert dist.num_classes == num_classes


class TestCreateSampleFix:
    def test_create_sample_preserves_distribution_type(self):
        d1 = ArrayProbabilityCategoricalDistribution(array=np.array([0.3, 0.7]))
        d2 = ArrayProbabilityCategoricalDistribution(array=np.array([0.6, 0.4]))
        d3 = ArrayProbabilityCategoricalDistribution(array=np.array([0.5, 0.5]))

        sample = create_sample([d1, d2, d3])

        assert isinstance(sample, ArrayCategoricalDistributionSample)
        assert isinstance(sample.array, ArrayCategoricalDistribution)

    def test_create_sample_shape(self):
        dists = [
            ArrayProbabilityCategoricalDistribution(array=np.array([0.2, 0.8])),
            ArrayProbabilityCategoricalDistribution(array=np.array([0.5, 0.5])),
        ]
        sample = create_sample(dists)

        # shape hides the protected class axis; only the batch (sample) dim is visible
        assert sample.shape == (2,)
        assert sample.sample_size == 2
        # The underlying array has the full shape (n_members, n_classes)
        assert sample.array.unnormalized_probabilities.shape == (2, 2)

    def test_create_sample_values_match(self):
        d1 = ArrayProbabilityCategoricalDistribution(array=np.array([0.3, 0.7]))
        d2 = ArrayProbabilityCategoricalDistribution(array=np.array([0.6, 0.4]))
        sample = create_sample([d1, d2])

        np.testing.assert_array_equal(
            sample.array.unnormalized_probabilities,
            np.array([[0.3, 0.7], [0.6, 0.4]]),
        )


class TestEndToEnd:
    def test_representer_produces_distribution_sample(self, trained_arf):
        arf, x = trained_arf
        sample = representer(arf).represent(x)

        assert isinstance(sample, ArrayCategoricalDistributionSample)

    def test_quantify_produces_decomposition(self, trained_arf):
        arf, x = trained_arf
        sample = representer(arf).represent(x)
        decomp = quantify(sample)

        assert hasattr(decomp, "total")
        assert hasattr(decomp, "aleatoric")
        assert hasattr(decomp, "epistemic")
        assert decomp.total >= 0
        assert decomp.aleatoric >= 0
        assert decomp.epistemic >= 0
        np.testing.assert_allclose(decomp.total, decomp.aleatoric + decomp.epistemic, atol=1e-10)


# ---------------------------------------------------------------------------
# ARFRegressor tests
# ---------------------------------------------------------------------------


@pytest.fixture
def trained_arf_regressor():
    """Train a small ARF regressor on Friedman stream and return (model, last_x)."""
    arf = ARFRegressor(n_models=5, seed=42)
    stream = synth.Friedman(seed=42)
    last_x: dict = {}
    for x, y in stream.take(200):
        arf.learn_one(x, y)
        last_x = x
    return arf, last_x


class TestRegressorCreateSample:
    def test_create_sample_produces_point_prediction_sample(self, trained_arf_regressor):
        arf, x = trained_arf_regressor
        result = predict_raw(arf, x)
        sample = create_sample(result)

        assert isinstance(sample, ArraySample)

    def test_sample_shape(self, trained_arf_regressor):
        arf, x = trained_arf_regressor
        result = predict_raw(arf, x)
        sample = create_sample(result)

        assert sample.array.shape == (1, arf.n_models)
        assert sample.sample_axis == 1


class TestRegressorEndToEnd:
    def test_representer_produces_point_prediction_sample(self, trained_arf_regressor):
        arf, x = trained_arf_regressor
        sample = representer(arf).represent(x)

        assert isinstance(sample, ArraySample)

    def test_measure_produces_variance(self, trained_arf_regressor):
        arf, x = trained_arf_regressor
        sample = representer(arf).represent(x)

        variance = measure(sample)
        quantify_res = quantify(sample)
        expected_variance = sample.sample_var()

        assert isinstance(variance, np.ndarray)
        assert isinstance(quantify_res, TotalDecomposition)

        np.testing.assert_allclose(variance, expected_variance, atol=1e-10)
        np.testing.assert_allclose(variance, quantify_res.total, atol=1e-10)

        assert np.all(np.asarray(variance) >= 0)
