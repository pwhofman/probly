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
        from probly.representation.distribution.array_categorical import (  # noqa: PLC0415
            ArrayLogitCategoricalDistribution,
            ArrayProbabilityCategoricalDistribution,
        )

        y_true = np.array([0, 1, 0])
        probs = np.array([[0.7, 0.3], [0.2, 0.8], [0.4, 0.6]])

        dist = ArrayProbabilityCategoricalDistribution(array=probs)
        assert float(accuracy(dist, y_true)) == pytest.approx(float(accuracy(probs, y_true)))
        assert float(expected_calibration_error(dist, y_true)) == pytest.approx(
            float(expected_calibration_error(probs, y_true))
        )
        assert float(classwise_ece(dist, y_true)) == pytest.approx(float(classwise_ece(probs, y_true)))

        # Logit-parameterized distributions must be normalized, not used raw.
        logit_dist = ArrayLogitCategoricalDistribution(array=np.log(probs))
        assert float(accuracy(logit_dist, y_true)) == pytest.approx(float(accuracy(probs, y_true)))
        assert float(expected_calibration_error(logit_dist, y_true)) == pytest.approx(
            float(expected_calibration_error(probs, y_true))
        )
        assert float(classwise_ece(logit_dist, y_true)) == pytest.approx(float(classwise_ece(probs, y_true)))


class TestReference(ReferenceSuite):
    pass
