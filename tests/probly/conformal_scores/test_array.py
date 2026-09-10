"""NumPy representation dispatch regressions."""

from __future__ import annotations

import numpy as np
import pytest

from probly.conformal_scores import (
    APSScore,
    SAPSScore,
    cqr_score,
    dirichlet_rl_score_func,
    inner_product_score_func,
    kl_divergence_score_func,
    tv_score_func,
    wasserstein_distance_score_func,
)
from probly.representation.distribution.array_categorical import (
    ArrayLogitCategoricalDistribution,
    ArrayProbabilityCategoricalDistribution,
)
from probly.representation.distribution.array_dirichlet import ArrayDirichletDistribution
from probly.representation.sample.array import ArraySample


@pytest.mark.parametrize(
    "score",
    [
        APSScore(randomized=False),
        SAPSScore(randomized=False, lambda_val=0.3),
        inner_product_score_func,
        kl_divergence_score_func,
        tv_score_func,
        wasserstein_distance_score_func,
    ],
)
@pytest.mark.parametrize("logits", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_categorical_scores(score, logits: bool, nested: bool) -> None:
    probabilities = np.array([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])
    labels = np.array([1, 2])
    distribution = (
        ArrayLogitCategoricalDistribution(np.log(probabilities) + 3.0)
        if logits
        else ArrayProbabilityCategoricalDistribution(probabilities * 5.0)
    )
    prediction = ArraySample(distribution, sample_axis=0) if nested else distribution
    np.testing.assert_allclose(score(prediction, labels), score(probabilities, labels), atol=1e-6)


def test_cqr_sample() -> None:
    intervals = np.array([[[0.0, 2.0], [1.0, 4.0]], [[1.0, 4.0], [2.0, 5.0]]])
    result = cqr_score(ArraySample(intervals, sample_axis=0), np.array([4.0, 0.0]))
    np.testing.assert_allclose(result, [1.0, 1.5])


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("sample_axis", [0, 1])
def test_dirichlet_relative_likelihood_samples(nested: bool, sample_axis: int) -> None:
    alphas = np.array([[[1.0, 2.0, 4.0], [3.0, 6.0, 2.0]], [[2.0, 4.0, 8.0], [6.0, 12.0, 4.0]]])
    labels = np.array([[1, 0], [2, 1]])
    values = ArrayDirichletDistribution(alphas) if nested else alphas
    sample = ArraySample(values, sample_axis=sample_axis, weights=np.array([0.25, 0.75]))
    result = dirichlet_rl_score_func(sample, labels)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [[0.5, 0.5], [0.0, 0.0]])
