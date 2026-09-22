"""Shared target-semantics checks for categorical distance scores."""

from __future__ import annotations

import numpy as np
import pytest

from probly.conformal_scores import (
    inner_product_score_func,
    kl_divergence_score_func,
    tv_score_func,
    wasserstein_distance_score_func,
)

SCORES = [inner_product_score_func, kl_divergence_score_func, tv_score_func, wasserstein_distance_score_func]
PREDICTIONS = np.array([[[0.2, 0.8], [0.7, 0.3]], [[0.4, 0.6], [0.9, 0.1]]])


def expected_score(score, predictions, probabilities):
    if score is inner_product_score_func:
        return 1 - np.sum(predictions * probabilities, axis=-1)
    if score is kl_divergence_score_func:
        return np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1) / predictions), axis=-1)
    if score is tv_score_func:
        return 0.5 * np.sum(np.abs(predictions - probabilities), axis=-1)
    return np.sum(np.abs(np.cumsum(predictions, axis=-1) - np.cumsum(probabilities, axis=-1)), axis=-1)


class ClassificationTargetSuite:
    @pytest.mark.parametrize("score", SCORES)
    @pytest.mark.parametrize("target", [[0.1, 0.9], [[0.1, 0.9], [0.7, 0.3]]])
    def test_broadcast_probability_targets(self, score, target, classification_backend):
        array, _, _ = classification_backend
        target = np.asarray(target)
        result = score(array(PREDICTIONS), array(target))
        np.testing.assert_allclose(result, expected_score(score, PREDICTIONS, target), atol=1e-6)

    @pytest.mark.parametrize("score", SCORES)
    @pytest.mark.parametrize("target", [1, [0, 1], [[0], [1]], [[0, 1], [1, 0]]])
    def test_integer_labels_broadcast_without_shape_inference(self, score, target, classification_backend):
        array, _, _ = classification_backend
        labels = np.asarray(target)
        result = score(array(PREDICTIONS), array(labels))
        np.testing.assert_allclose(result, expected_score(score, PREDICTIONS, np.eye(2)[labels]), atol=1e-6)

    @pytest.mark.parametrize("score", SCORES)
    @pytest.mark.parametrize("dtype", [int, float])
    def test_dtype_alone_selects_label_or_distribution(self, score, dtype, classification_backend):
        array, _, _ = classification_backend
        targets = np.array([[0, 1], [1, 0]], dtype=dtype)
        probabilities = np.eye(2)[targets] if dtype is int else targets
        result = score(array(PREDICTIONS), array(targets))
        np.testing.assert_allclose(result, expected_score(score, PREDICTIONS, probabilities), atol=1e-6)

    @pytest.mark.parametrize("score", SCORES)
    def test_unbatched_predictions(self, score, classification_backend):
        array, _, _ = classification_backend
        predictions = np.array([0.2, 0.8])
        for target in (np.array(1), np.array([0.0, 1.0])):
            result = score(array(predictions), array(target))
            np.testing.assert_allclose(result, expected_score(score, predictions, np.array([0.0, 1.0])), atol=1e-6)

    @pytest.mark.parametrize("score", SCORES)
    @pytest.mark.parametrize("logits", [False, True])
    def test_categorical_targets_use_normalized_probabilities(self, score, logits, classification_backend):
        array, probability_distribution, logit_distribution = classification_backend
        target_values = np.array([[1, 9], [7, 3]])
        targets = (
            logit_distribution(array(np.log(target_values) + 3))
            if logits
            else probability_distribution(array(target_values))
        )
        expected = expected_score(score, PREDICTIONS, target_values / 10)
        for predictions in (array(PREDICTIONS), probability_distribution(array(PREDICTIONS))):
            np.testing.assert_allclose(score(predictions, targets), expected, atol=1e-6)

    @pytest.mark.parametrize("score", SCORES)
    @pytest.mark.parametrize("target", [np.array([True, False]), np.array([0.2 + 0j, 0.8 + 0j])])
    def test_unsupported_target_dtype_is_rejected(self, score, target, classification_backend):
        array, _, _ = classification_backend
        with pytest.raises(TypeError, match="Targets must be"):
            score(array(PREDICTIONS), array(target))

    @pytest.mark.parametrize("score", SCORES)
    @pytest.mark.parametrize("target", [np.array(1.0), np.array([1.0]), np.array([0.2, 0.3, 0.5])])
    def test_probability_class_axis_is_validated(self, score, target, classification_backend):
        array, _, _ = classification_backend
        with pytest.raises(ValueError, match="Target probabilities"):
            score(array(PREDICTIONS), array(target))
