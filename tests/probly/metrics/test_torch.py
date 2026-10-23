"""PyTorch backend tests for probly.metrics."""

from __future__ import annotations

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

torch = pytest.importorskip("torch")


@pytest.fixture
def array_fn():
    return torch.tensor


@pytest.fixture
def array_type():
    return torch.Tensor


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


class TestGradientTransparency:
    """Metrics neither detach their inputs nor disable autograd."""

    def test_expected_calibration_error_backward(self):
        from probly.metrics import expected_calibration_error  # noqa: PLC0415

        probs = torch.softmax(torch.randn(64, 4), dim=1).requires_grad_()
        labels = torch.randint(0, 4, (64,))
        expected_calibration_error(probs, labels).backward()
        assert probs.grad is not None
        assert probs.grad.abs().sum() > 0


class TestDistributionInput:
    """Probability-based metrics accept categorical distribution predictions."""

    def test_metrics_unwrap_categorical_distributions(self):
        from probly.metrics import accuracy, classwise_ece, expected_calibration_error  # noqa: PLC0415
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchLogitCategoricalDistribution,
            TorchProbabilityCategoricalDistribution,
        )

        y_true = torch.tensor([0, 1, 0])
        probs = torch.tensor([[0.7, 0.3], [0.2, 0.8], [0.4, 0.6]])

        dist = TorchProbabilityCategoricalDistribution(tensor=probs)
        assert float(accuracy(dist, y_true)) == pytest.approx(float(accuracy(probs, y_true)))
        assert float(expected_calibration_error(dist, y_true)) == pytest.approx(
            float(expected_calibration_error(probs, y_true))
        )
        assert float(classwise_ece(dist, y_true)) == pytest.approx(float(classwise_ece(probs, y_true)))

        # Logit-parameterized distributions must be normalized, not used raw.
        logit_dist = TorchLogitCategoricalDistribution(tensor=torch.log(probs))
        assert float(accuracy(logit_dist, y_true)) == pytest.approx(float(accuracy(probs, y_true)))
        assert float(expected_calibration_error(logit_dist, y_true)) == pytest.approx(
            float(expected_calibration_error(probs, y_true)), abs=1e-6
        )
        assert float(classwise_ece(logit_dist, y_true)) == pytest.approx(float(classwise_ece(probs, y_true)), abs=1e-6)


class TestReference(ReferenceSuite):
    pass


class TestScoringRules(ScoringRuleSuite):
    pass


class TestScoringRuleInputs:
    """Scoring rules accept categorical distributions and reject samples."""

    def test_log_loss_of_logit_distribution_is_exact(self):
        from probly.metrics import log_loss_per_instance  # noqa: PLC0415
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchLogitCategoricalDistribution,
        )

        # exp(-800) underflows to zero, so only the log probabilities keep this loss finite.
        dist = TorchLogitCategoricalDistribution(tensor=torch.tensor([[0.0, -800.0]]))
        assert float(log_loss_per_instance(dist, torch.tensor([1]))[0]) == pytest.approx(800.0)

    def test_logit_distribution_scores_normalized_probabilities(self):
        from probly.metrics import (  # noqa: PLC0415
            brier_loss_per_instance,
            log_loss_per_instance,
            spherical_loss_per_instance,
            zero_one_loss_per_instance,
        )
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchLogitCategoricalDistribution,
        )

        probs = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=torch.float64)
        labels = torch.tensor([0, 1])
        # Shifted log probabilities are unnormalized logits of the same distribution.
        dist = TorchLogitCategoricalDistribution(tensor=torch.log(probs) + 1.0)
        for rule in (
            log_loss_per_instance,
            brier_loss_per_instance,
            zero_one_loss_per_instance,
            spherical_loss_per_instance,
        ):
            torch.testing.assert_close(rule(dist, labels), rule(probs, labels), msg=rule.__name__)

    def test_distribution_target_matches_array(self):
        from probly.metrics import brier_loss_per_instance  # noqa: PLC0415
        from probly.representation.distribution.torch_categorical import (  # noqa: PLC0415
            TorchProbabilityCategoricalDistribution,
        )

        probs = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
        target = torch.tensor([[0.5, 0.5], [0.1, 0.9]])
        result = brier_loss_per_instance(probs, TorchProbabilityCategoricalDistribution(tensor=target))
        torch.testing.assert_close(result, brier_loss_per_instance(probs, target))

    def test_rejects_samples(self):
        from probly.metrics import brier_loss_per_instance  # noqa: PLC0415
        from probly.representation.sample.torch import TorchSample  # noqa: PLC0415

        sample = TorchSample(torch.tensor([[[0.7, 0.3]], [[0.4, 0.6]]]), sample_dim=0)
        with pytest.raises(TypeError, match="sample_mean"):
            brier_loss_per_instance(sample, torch.tensor([0]))

    def test_backward(self):
        from probly.metrics import log_loss_per_instance  # noqa: PLC0415

        probs = torch.softmax(torch.randn(8, 3), dim=1).requires_grad_()
        log_loss_per_instance(probs, torch.randint(0, 3, (8,))).sum().backward()
        assert probs.grad is not None
        assert torch.isfinite(probs.grad).all()
