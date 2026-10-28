"""JAX backend tests for probly.metrics."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
from jax.experimental import checkify  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from probly import metrics  # noqa: E402

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
from ._scoring_rule_suite import ScoringRuleSuite  # noqa: E402


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
        from probly.representation.distribution.numpy_categorical import (  # noqa: PLC0415
            NumpyProbabilityCategoricalDistribution,
        )

        y_true = jnp.array([0, 1, 0])
        probs = jnp.array([[0.7, 0.3], [0.2, 0.8], [0.4, 0.6]])

        # Numpy-backed distributions are what flax prediction pipelines produce.
        dist = NumpyProbabilityCategoricalDistribution(array=np.asarray(probs))
        assert float(accuracy(dist, y_true)) == pytest.approx(float(accuracy(probs, y_true)))
        assert float(expected_calibration_error(dist, y_true)) == pytest.approx(
            float(expected_calibration_error(probs, y_true)), abs=1e-6
        )
        assert float(classwise_ece(dist, y_true)) == pytest.approx(float(classwise_ece(probs, y_true)), abs=1e-6)


class TestReference(ReferenceSuite):
    pass


class TestScoringRules(ScoringRuleSuite):
    pass


class TestScoringRuleInputs:
    """Scoring rules accept categorical distributions and reject samples."""

    def test_log_loss_of_logit_distribution_is_exact(self):
        from probly.metrics import log_loss_per_instance  # noqa: PLC0415
        from probly.representation.distribution.jax_categorical import (  # noqa: PLC0415
            JaxLogitCategoricalDistribution,
        )

        # exp(-800) underflows to zero, so only the log probabilities keep this loss finite.
        dist = JaxLogitCategoricalDistribution(jnp.array([[0.0, -800.0]]))
        assert float(log_loss_per_instance(dist, jnp.array([1]))[0]) == pytest.approx(800.0)

    def test_logit_distribution_scores_normalized_probabilities(self):
        from probly.metrics import (  # noqa: PLC0415
            brier_loss_per_instance,
            log_loss_per_instance,
            spherical_loss_per_instance,
            zero_one_loss_per_instance,
        )
        from probly.representation.distribution.jax_categorical import (  # noqa: PLC0415
            JaxLogitCategoricalDistribution,
        )

        probs = jnp.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
        labels = jnp.array([0, 1])
        # Shifted log probabilities are unnormalized logits of the same distribution.
        dist = JaxLogitCategoricalDistribution(jnp.log(probs) + 1.0)
        for rule in (
            log_loss_per_instance,
            brier_loss_per_instance,
            zero_one_loss_per_instance,
            spherical_loss_per_instance,
        ):
            assert jnp.allclose(rule(dist, labels), rule(probs, labels), rtol=1e-5), rule.__name__

    def test_distribution_target_matches_array(self):
        from probly.metrics import brier_loss_per_instance  # noqa: PLC0415
        from probly.representation.distribution.jax_categorical import (  # noqa: PLC0415
            JaxProbabilityCategoricalDistribution,
        )

        probs = jnp.array([[0.7, 0.3], [0.2, 0.8]])
        target = jnp.array([[0.5, 0.5], [0.1, 0.9]])
        result = brier_loss_per_instance(probs, JaxProbabilityCategoricalDistribution(target))
        assert jnp.allclose(result, brier_loss_per_instance(probs, target))

    def test_rejects_samples(self):
        from probly.metrics import brier_loss_per_instance  # noqa: PLC0415
        from probly.representation.sample.jax import JaxSample  # noqa: PLC0415

        sample = JaxSample(jnp.array([[[0.7, 0.3]], [[0.4, 0.6]]]), sample_axis=0)
        with pytest.raises(TypeError, match="sample_mean"):
            brier_loss_per_instance(sample, jnp.array([0]))

    @pytest.mark.parametrize("representation", ["array", "probability", "logit"])
    @pytest.mark.parametrize("target_kind", ["labels", "probabilities", "distribution"])
    def test_log_loss_backward_with_zero_probability(self, representation, target_kind):
        from probly.representation.distribution.jax_categorical import (  # noqa: PLC0415
            JaxLogitCategoricalDistribution,
            JaxProbabilityCategoricalDistribution,
        )

        def loss(values):
            predictions = values
            if representation == "probability":
                predictions = JaxProbabilityCategoricalDistribution(values)
            elif representation == "logit":
                predictions = JaxLogitCategoricalDistribution(values)
            target = jnp.array([0]) if target_kind == "labels" else jnp.array([[1.0, 0.0]])
            if target_kind == "distribution":
                target = JaxProbabilityCategoricalDistribution(target)
            return metrics.log_loss(predictions, target)

        values = jnp.array([[0.0, -1000.0]] if representation == "logit" else [[1.0, 0.0]])
        error, (actual_loss, gradient) = jax.jit(checkify.checkify(jax.value_and_grad(loss)))(values)
        error.throw()
        assert float(actual_loss) == 0.0
        expected = {"array": [[-1.0, 0.0]], "probability": [[0.0, 1.0]], "logit": [[0.0, 0.0]]}
        assert jnp.allclose(gradient, jnp.array(expected[representation]))

    @pytest.mark.parametrize("soft_target", [False, True])
    def test_log_loss_backward_through_underflowed_softmax(self, soft_target):
        target = jnp.array([[1.0, 0.0]]) if soft_target else jnp.array([0])
        gradient = jax.jit(jax.grad(lambda logits: metrics.log_loss(jax.nn.softmax(logits), target)))(
            jnp.array([[0.0, -1000.0]])
        )
        assert jnp.allclose(gradient, 0.0)

    @pytest.mark.parametrize("soft_target", [False, True])
    def test_extreme_logits_have_exact_value_and_gradient(self, soft_target):
        from probly.representation.distribution.jax_categorical import (  # noqa: PLC0415
            JaxLogitCategoricalDistribution,
        )

        target = jnp.array([[0.3, 0.7]]) if soft_target else jnp.array([1])
        value, gradient = jax.jit(
            jax.value_and_grad(lambda logits: metrics.log_loss(JaxLogitCategoricalDistribution(logits), target))
        )(jnp.array([[0.0, -1000.0]]))
        assert float(value) == pytest.approx(700.0 if soft_target else 1000.0)
        expected = jnp.array([[0.7, -0.7]] if soft_target else [[1.0, -1.0]])
        assert jnp.allclose(gradient, expected)


class TestTransforms:
    @pytest.mark.parametrize(
        "rule",
        [
            metrics.log_loss_per_instance,
            metrics.brier_loss_per_instance,
            metrics.spherical_loss_per_instance,
            metrics.zero_one_loss_per_instance,
            metrics.log_loss,
            metrics.brier_loss,
            metrics.spherical_loss,
            metrics.zero_one_loss,
        ],
    )
    def test_scoring_rules_support_jit_vmap_and_grad(self, rule):
        probabilities = jnp.array([[0.7, 0.3], [0.2, 0.8]])
        labels = jnp.array([0, 1])
        assert jnp.allclose(jax.jit(rule)(probabilities, labels), rule(probabilities, labels))
        mapped = jax.jit(jax.vmap(rule))(probabilities, labels)
        expected = jnp.stack([rule(p, y) for p, y in zip(probabilities, labels, strict=True)])
        assert jnp.allclose(mapped, expected)
        gradient = jax.jit(jax.grad(lambda p: jnp.sum(rule(p, labels))))(probabilities)
        assert jnp.isfinite(gradient).all()

    @pytest.mark.parametrize(
        "metric",
        [metrics.accuracy, metrics.classwise_ece, metrics.expected_calibration_error],
    )
    def test_classification_metrics_support_jit(self, metric):
        probabilities = jnp.array([[0.7, 0.3], [0.2, 0.8]])
        labels = jnp.array([0, 1])
        assert jnp.allclose(jax.jit(metric)(probabilities, labels), metric(probabilities, labels))

    @pytest.mark.parametrize(
        "metric",
        [
            metrics.auc,
            metrics.false_positive_rate,
            metrics.false_negative_rate,
            metrics.roc_curve,
            metrics.precision_recall_curve,
            metrics.roc_auc_score,
            metrics.average_precision_score,
        ],
    )
    def test_binary_metrics_support_jit(self, metric):
        first = jnp.array([0.0, 1.0, 0.0, 1.0])
        second = jnp.array([0.0, 1.0, 1.0, 1.0])
        actual = jax.jit(metric)(first, second)
        expected = metric(first, second)
        assert all(
            jnp.allclose(a, e, equal_nan=True)
            for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True)
        )
