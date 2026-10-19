"""NumPy backend tests for probly.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from ._accuracy_suite import AccuracySuite
from ._ap_score_suite import APScoreSuite
from ._auc_suite import AUCSuite
from ._classwise_ece_suite import ClasswiseECESuite
from ._error_rate_suite import ErrorRateSuite
from ._expected_calibration_error_suite import ExpectedCalibrationErrorSuite
from ._pr_curve_suite import PRCurveSuite
from ._reference_suite import ReferenceSuite
from ._roc_auc_score_suite import RocAucScoreSuite
from ._roc_curve_suite import RocCurveSuite
from ._scoring_rule_suite import ScoringRuleSuite


@pytest.fixture
def array_fn():
    return np.array


@pytest.fixture
def array_type():
    return np.generic


class TestAUC(AUCSuite):
    pass


class TestRocCurve(RocCurveSuite):
    pass


class TestPRCurve(PRCurveSuite):
    pass


class TestRocAucScore(RocAucScoreSuite):
    pass


class TestAPScore(APScoreSuite):
    pass


class TestClasswiseECE(ClasswiseECESuite):
    pass


class TestAccuracy(AccuracySuite):
    pass


class TestExpectedCalibrationError(ExpectedCalibrationErrorSuite):
    pass


class TestErrorRates(ErrorRateSuite):
    pass


class TestDistributionInput:
    """Probability-based metrics accept categorical distribution predictions."""

    def test_metrics_unwrap_categorical_distributions(self):
        from probly.metrics import accuracy, classwise_ece, expected_calibration_error  # noqa: PLC0415
        from probly.representation.distribution.numpy_categorical import (  # noqa: PLC0415
            NumpyLogitCategoricalDistribution,
            NumpyProbabilityCategoricalDistribution,
        )

        y_true = np.array([0, 1, 0])
        probs = np.array([[0.7, 0.3], [0.2, 0.8], [0.4, 0.6]])

        dist = NumpyProbabilityCategoricalDistribution(array=probs)
        assert float(accuracy(dist, y_true)) == pytest.approx(float(accuracy(probs, y_true)))
        assert float(expected_calibration_error(dist, y_true)) == pytest.approx(
            float(expected_calibration_error(probs, y_true))
        )
        assert float(classwise_ece(dist, y_true)) == pytest.approx(float(classwise_ece(probs, y_true)))

        # Logit-parameterized distributions must be normalized, not used raw.
        logit_dist = NumpyLogitCategoricalDistribution(array=np.log(probs))
        assert float(accuracy(logit_dist, y_true)) == pytest.approx(float(accuracy(probs, y_true)))
        assert float(expected_calibration_error(logit_dist, y_true)) == pytest.approx(
            float(expected_calibration_error(probs, y_true))
        )
        assert float(classwise_ece(logit_dist, y_true)) == pytest.approx(float(classwise_ece(probs, y_true)))


class TestReference(ReferenceSuite):
    pass


class TestScoringRules(ScoringRuleSuite):
    pass


class TestScoringRuleInputs:
    """Scoring rules accept categorical distributions and reject samples."""

    def test_log_loss_of_logit_distribution_is_exact(self):
        from probly.metrics import log_loss_per_instance  # noqa: PLC0415
        from probly.representation.distribution.numpy_categorical import (  # noqa: PLC0415
            NumpyLogitCategoricalDistribution,
        )

        # exp(-800) underflows to zero, so only the log probabilities keep this loss finite.
        dist = NumpyLogitCategoricalDistribution(array=np.array([[0.0, -800.0]]))
        assert float(log_loss_per_instance(dist, np.array([1]))[0]) == pytest.approx(800.0)

    def test_logit_distribution_scores_normalized_probabilities(self):
        from probly.metrics import (  # noqa: PLC0415
            brier_loss_per_instance,
            log_loss_per_instance,
            spherical_loss_per_instance,
            zero_one_loss_per_instance,
        )
        from probly.representation.distribution.numpy_categorical import (  # noqa: PLC0415
            NumpyLogitCategoricalDistribution,
        )

        probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
        labels = np.array([0, 1])
        # Shifted log probabilities are unnormalized logits of the same distribution.
        dist = NumpyLogitCategoricalDistribution(array=np.log(probs) + 1.0)
        for rule in (
            log_loss_per_instance,
            brier_loss_per_instance,
            zero_one_loss_per_instance,
            spherical_loss_per_instance,
        ):
            np.testing.assert_allclose(rule(dist, labels), rule(probs, labels), rtol=1e-12, err_msg=rule.__name__)

    def test_distribution_target_matches_array(self):
        from probly.metrics import brier_loss_per_instance  # noqa: PLC0415
        from probly.representation.distribution.numpy_categorical import (  # noqa: PLC0415
            NumpyProbabilityCategoricalDistribution,
        )

        probs = np.array([[0.7, 0.3], [0.2, 0.8]])
        target = np.array([[0.5, 0.5], [0.1, 0.9]])
        result = brier_loss_per_instance(probs, NumpyProbabilityCategoricalDistribution(array=target))
        np.testing.assert_allclose(result, brier_loss_per_instance(probs, target))

    def test_rejects_samples(self):
        from probly.metrics import brier_loss_per_instance  # noqa: PLC0415
        from probly.representation.sample.numpy import NumpySample  # noqa: PLC0415

        sample = NumpySample(np.array([[[0.7, 0.3]], [[0.4, 0.6]]]), sample_axis=0)
        with pytest.raises(TypeError, match="sample_mean"):
            brier_loss_per_instance(sample, np.array([0]))
