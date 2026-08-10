"""JAX backend tests for probly.metrics."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from ._accuracy_suite import AccuracySuite  # noqa: E402
from ._ap_score_suite import APScoreSuite  # noqa: E402
from ._auc_suite import AUCSuite  # noqa: E402
from ._classwise_ece_suite import ClasswiseECESuite  # noqa: E402
from ._error_rate_suite import ErrorRateSuite  # noqa: E402
from ._expected_calibration_error_suite import ExpectedCalibrationErrorSuite  # noqa: E402
from ._pr_curve_suite import PRCurveSuite  # noqa: E402
from ._reference_suite import ReferenceSuite  # noqa: E402
from ._roc_auc_score_suite import RocAucScoreSuite  # noqa: E402
from ._roc_curve_suite import RocCurveSuite  # noqa: E402


@pytest.fixture
def array_fn():
    return jnp.array


@pytest.fixture
def array_type():
    return jax.Array


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
        import numpy as np  # noqa: PLC0415

        from probly.metrics import accuracy, classwise_ece, expected_calibration_error  # noqa: PLC0415
        from probly.representation.distribution.array_categorical import (  # noqa: PLC0415
            ArrayProbabilityCategoricalDistribution,
        )

        y_true = jnp.array([0, 1, 0])
        probs = jnp.array([[0.7, 0.3], [0.2, 0.8], [0.4, 0.6]])

        # Numpy-backed distributions are what flax prediction pipelines produce.
        dist = ArrayProbabilityCategoricalDistribution(array=np.asarray(probs))
        assert float(accuracy(dist, y_true)) == pytest.approx(float(accuracy(probs, y_true)))
        assert float(expected_calibration_error(dist, y_true)) == pytest.approx(
            float(expected_calibration_error(probs, y_true)), abs=1e-6
        )
        assert float(classwise_ece(dist, y_true)) == pytest.approx(float(classwise_ece(probs, y_true)), abs=1e-6)


class TestReference(ReferenceSuite):
    pass
