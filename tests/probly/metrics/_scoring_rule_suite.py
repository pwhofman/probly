"""Scoring rule test suite."""

from __future__ import annotations

import math

import numpy as np
import pytest

from probly.metrics import (
    accuracy,
    brier_loss,
    brier_loss_per_instance,
    log_loss,
    log_loss_per_instance,
    spherical_loss,
    spherical_loss_per_instance,
    zero_one_loss,
    zero_one_loss_per_instance,
)

_PROBS = [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]
_LABELS = [0, 1]
_RULES = [log_loss_per_instance, brier_loss_per_instance, zero_one_loss_per_instance, spherical_loss_per_instance]
_AVERAGED = [
    (log_loss, log_loss_per_instance),
    (brier_loss, brier_loss_per_instance),
    (zero_one_loss, zero_one_loss_per_instance),
    (spherical_loss, spherical_loss_per_instance),
]


class ScoringRuleSuite:
    """Test suite for the per-instance scoring rules."""

    @pytest.mark.parametrize(
        ("rule", "expected"),
        [
            (log_loss_per_instance, [-math.log(0.7), -math.log(0.3)]),
            (brier_loss_per_instance, [0.3**2 + 0.2**2 + 0.1**2, 0.1**2 + 0.7**2 + 0.6**2]),
            (zero_one_loss_per_instance, [0.0, 1.0]),
            (spherical_loss_per_instance, [1 - 0.7 / math.sqrt(0.54), 1 - 0.3 / math.sqrt(0.46)]),
        ],
    )
    def test_known_values(self, array_fn, rule, expected):
        """Integer labels give the loss of the true class for each instance."""
        result = rule(array_fn(_PROBS), array_fn(_LABELS))
        assert np.asarray(result, dtype=float).tolist() == pytest.approx(expected, rel=1e-5)

    @pytest.mark.parametrize("rule", _RULES)
    def test_one_hot_target_matches_labels(self, array_fn, rule):
        """A one-hot target distribution gives the same loss as the equivalent integer labels."""
        one_hot = np.eye(3)[_LABELS].tolist()
        from_labels = rule(array_fn(_PROBS), array_fn(_LABELS))
        from_target = rule(array_fn(_PROBS), array_fn(one_hot))
        np.testing.assert_allclose(np.asarray(from_target), np.asarray(from_labels), rtol=1e-6)

    @pytest.mark.parametrize("rule", _RULES)
    def test_target_probabilities_give_expected_loss(self, array_fn, rule):
        """Target probabilities weight the loss of each class by its target mass."""
        target = [[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]]
        per_class = [np.asarray(rule(array_fn(_PROBS), array_fn([k, k]))) for k in range(3)]
        expected = sum(np.asarray(target)[:, k] * per_class[k] for k in range(3))
        result = rule(array_fn(_PROBS), array_fn(target))
        np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-6)

    @pytest.mark.parametrize("rule", _RULES)
    def test_preserves_batch_shape(self, array_fn, rule):
        """One loss is returned per instance for any number of batch dimensions."""
        probs = array_fn(np.full((4, 3, 5), 0.2))
        labels = array_fn(np.zeros((4, 3), dtype=np.int64))
        assert tuple(rule(probs, labels).shape) == (4, 3)

    def test_log_loss_is_infinite_without_clipping(self, array_fn):
        """A zero probability on the true class gives an infinite log loss."""
        result = log_loss_per_instance(array_fn([[1.0, 0.0]]), array_fn([1]))
        assert np.isposinf(np.asarray(result)).all()

    def test_log_loss_ignores_classes_without_target_mass(self, array_fn):
        """A class the target never puts mass on adds no loss, even where its log loss is infinite."""
        result = log_loss_per_instance(array_fn([[1.0, 0.0]]), array_fn([[1.0, 0.0]]))
        assert np.asarray(result).tolist() == [0.0]

    def test_rejects_mismatched_target_shape(self, array_fn):
        """Targets must be labels of the batch shape or probabilities of the prediction shape."""
        with pytest.raises(ValueError, match="y_true must be"):
            brier_loss_per_instance(array_fn(_PROBS), array_fn([[0, 1, 2]]))

    def test_rejects_float_labels(self, array_fn):
        """Labels of the batch shape must be integers rather than being cast silently."""
        with pytest.raises(ValueError, match="integer"):
            brier_loss_per_instance(array_fn(_PROBS), array_fn([0.0, 1.0]))

    @pytest.mark.parametrize(("metric", "rule"), _AVERAGED)
    def test_averaged_metric_is_mean_of_per_instance(self, array_fn, array_type, metric, rule):
        """Each averaged metric returns the backend's scalar mean of its per-instance rule."""
        result = metric(array_fn(_PROBS), array_fn(_LABELS))
        assert isinstance(result, array_type)
        expected = np.asarray(rule(array_fn(_PROBS), array_fn(_LABELS))).mean()
        assert float(result) == pytest.approx(float(expected), rel=1e-6)

    @pytest.mark.parametrize(
        "probs",
        [
            [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.2, 0.5, 0.3]],
            # Exact ties, where argmax conventions decide whether a prediction counts as correct.
            [[0.4, 0.4, 0.2], [0.2, 0.4, 0.4], [1 / 3, 1 / 3, 1 / 3]],
        ],
    )
    def test_zero_one_loss_is_one_minus_accuracy(self, array_fn, probs):
        """The independently implemented accuracy and zero-one loss agree, including on ties."""
        labels = array_fn([1, 1, 0])
        assert float(zero_one_loss(array_fn(probs), labels)) == pytest.approx(
            1.0 - float(accuracy(array_fn(probs), labels))
        )
