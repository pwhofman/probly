"""Tests for the small helper functions in ``transformation.calibration.flax``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import jax as jax_types


def _flax_modules():
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    return jax, jnp


class TestReshapeBinary:
    def test_squeezes_trailing_singleton(self) -> None:
        _, jnp = _flax_modules()
        from probly.transformation.calibration.flax import _reshape_binary_preds  # noqa: PLC0415

        preds = jnp.zeros((3, 1))
        out = _reshape_binary_preds(preds)
        assert out.shape == (3,)

    def test_keeps_higher_dim(self) -> None:
        _, jnp = _flax_modules()
        from probly.transformation.calibration.flax import _reshape_binary_preds  # noqa: PLC0415

        preds = jnp.zeros((3,))
        out = _reshape_binary_preds(preds)
        assert out.shape == (3,)


class TestReshapeBinaryLabels:
    def test_match(self) -> None:
        _, jnp = _flax_modules()
        from probly.transformation.calibration.flax import _reshape_binary_labels  # noqa: PLC0415

        labels = jnp.array([[0, 1], [1, 0]])
        out = _reshape_binary_labels(labels, expected_elements=4)
        assert out.shape == (4,)

    def test_mismatch_raises(self) -> None:
        _, jnp = _flax_modules()
        from probly.transformation.calibration.flax import _reshape_binary_labels  # noqa: PLC0415

        with pytest.raises(ValueError, match="Binary calibration labels"):
            _reshape_binary_labels(jnp.array([0, 1]), expected_elements=3)


class TestReshapeMulticlassLabels:
    def test_match(self) -> None:
        _, jnp = _flax_modules()
        from probly.transformation.calibration.flax import _reshape_multiclass_labels  # noqa: PLC0415

        labels = jnp.array([[0, 1], [1, 0]])
        out = _reshape_multiclass_labels(labels, batch_shape=(2, 2))
        assert out.shape == (4,)

    def test_mismatch_raises(self) -> None:
        _, jnp = _flax_modules()
        from probly.transformation.calibration.flax import _reshape_multiclass_labels  # noqa: PLC0415

        with pytest.raises(ValueError, match="Multiclass calibration labels"):
            _reshape_multiclass_labels(jnp.array([0]), batch_shape=(2,))

    def test_empty_batch_shape(self) -> None:
        _, jnp = _flax_modules()
        from probly.transformation.calibration.flax import _reshape_multiclass_labels  # noqa: PLC0415

        out = _reshape_multiclass_labels(jnp.array([3]), batch_shape=())
        assert out.shape == (1,)


class TestCalibrationNll:
    def test_multiclass_loss(self) -> None:
        _, jnp = _flax_modules()
        from probly.transformation.calibration.flax import _calibration_nll  # noqa: PLC0415

        scaled_logits = jnp.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        labels = jnp.array([2, 0])
        loss = _calibration_nll(scaled_logits, labels)
        assert bool(jnp.isfinite(loss))
        assert float(loss) > 0

    def test_binary_loss(self) -> None:
        _, jnp = _flax_modules()
        from probly.transformation.calibration.flax import _calibration_nll  # noqa: PLC0415

        scaled_logits = jnp.array([0.5, -0.5, 1.0])
        labels = jnp.array([1, 0, 1])
        loss = _calibration_nll(scaled_logits, labels)
        assert bool(jnp.isfinite(loss))


class TestPrepareBinaryIsotonicInputs:
    def test_returns_flat_inputs(self) -> None:
        _, jnp = _flax_modules()
        from probly.transformation.calibration.flax import _prepare_binary_isotonic_inputs  # noqa: PLC0415

        preds = jnp.zeros(5)
        labels = jnp.zeros(5)
        flat_logits, flat_labels, singleton = _prepare_binary_isotonic_inputs(preds, labels)
        assert flat_logits.shape == (5,)
        assert flat_labels.shape == (5,)
        assert singleton is False

    def test_singleton_class_axis(self) -> None:
        _, jnp = _flax_modules()
        from probly.transformation.calibration.flax import _prepare_binary_isotonic_inputs  # noqa: PLC0415

        preds = jnp.zeros((5, 1))
        labels = jnp.zeros(5)
        _, _, singleton = _prepare_binary_isotonic_inputs(preds, labels)
        assert singleton is True

    def test_zero_dim_raises(self) -> None:
        _, jnp = _flax_modules()
        from probly.transformation.calibration.flax import _prepare_binary_isotonic_inputs  # noqa: PLC0415

        with pytest.raises(ValueError, match="at least one dimension"):
            _prepare_binary_isotonic_inputs(jnp.array(0.5), jnp.array(0.0))

    def test_multiclass_raises(self) -> None:
        _, jnp = _flax_modules()
        from probly.transformation.calibration.flax import _prepare_binary_isotonic_inputs  # noqa: PLC0415

        with pytest.raises(ValueError, match="binary logits only"):
            _prepare_binary_isotonic_inputs(jnp.zeros((5, 3)), jnp.zeros(5))

    def test_shape_mismatch_raises(self) -> None:
        _, jnp = _flax_modules()
        from probly.transformation.calibration.flax import _prepare_binary_isotonic_inputs  # noqa: PLC0415

        with pytest.raises(ValueError, match="batch size"):
            _prepare_binary_isotonic_inputs(jnp.zeros(5), jnp.zeros(3))


class TestApplyAffine:
    def test_scalar_temperature(self) -> None:
        _, jnp = _flax_modules()
        from probly.transformation.calibration.flax import _apply_affine  # noqa: PLC0415

        logits = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        temperature = jnp.array([2.0])
        out = _apply_affine(logits, temperature, bias=None)
        assert jnp.allclose(out, logits / 2.0)

    def test_per_class_temperature(self) -> None:
        _, jnp = _flax_modules()
        from probly.transformation.calibration.flax import _apply_affine  # noqa: PLC0415

        logits = jnp.array([[1.0, 4.0]])
        temperature = jnp.array([1.0, 2.0])
        out = _apply_affine(logits, temperature, bias=None)
        assert jnp.allclose(out, jnp.array([[1.0, 2.0]]))

    def test_with_scalar_bias(self) -> None:
        _, jnp = _flax_modules()
        from probly.transformation.calibration.flax import _apply_affine  # noqa: PLC0415

        logits = jnp.array([[1.0, 2.0]])
        temperature = jnp.array([1.0])
        bias = jnp.array([0.5])
        out = _apply_affine(logits, temperature, bias=bias)
        assert jnp.allclose(out, jnp.array([[1.5, 2.5]]))

    def test_with_per_class_bias(self) -> None:
        _, jnp = _flax_modules()
        from probly.transformation.calibration.flax import _apply_affine  # noqa: PLC0415

        logits = jnp.array([[1.0, 2.0]])
        temperature = jnp.array([1.0, 1.0])
        bias = jnp.array([0.1, 0.2])
        out = _apply_affine(logits, temperature, bias=bias)
        assert jnp.allclose(out, jnp.array([[1.1, 2.2]]))

    def test_zero_dim_raises(self) -> None:
        _, jnp = _flax_modules()
        from probly.transformation.calibration.flax import _apply_affine  # noqa: PLC0415

        with pytest.raises(ValueError, match="at least one dimension"):
            _apply_affine(jnp.array(0.5), jnp.array([1.0]), None)

    def test_temperature_shape_mismatch_raises(self) -> None:
        _, jnp = _flax_modules()
        from probly.transformation.calibration.flax import _apply_affine  # noqa: PLC0415

        logits = jnp.array([[1.0, 2.0]])
        temperature = jnp.array([1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="Temperature"):
            _apply_affine(logits, temperature, None)

    def test_bias_shape_mismatch_raises(self) -> None:
        _, jnp = _flax_modules()
        from probly.transformation.calibration.flax import _apply_affine  # noqa: PLC0415

        logits = jnp.array([[1.0, 2.0]])
        temperature = jnp.array([1.0, 1.0])
        bias = jnp.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="Bias"):
            _apply_affine(logits, temperature, bias)


class TestFlaxIdentityLogitModel:
    def test_call_returns_unchanged(self) -> None:
        _, jnp = _flax_modules()
        from probly.transformation.calibration.flax import FlaxIdentityLogitModel  # noqa: PLC0415

        model = FlaxIdentityLogitModel()
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        out = model(x)
        assert jnp.array_equal(out, x)


class TestFlaxCalibrationPredictorBaseHelpers:
    def test_fit_swaps_argument_order(self) -> None:
        """``fit(x, y)`` reorders args to call ``calibrate(y, x)``."""
        jax, _ = _flax_modules()
        from probly.method.calibration import flax_identity_logit_model, temperature_scaling  # noqa: PLC0415
        from probly.predictor import predict_raw  # noqa: PLC0415

        wrapper = temperature_scaling(flax_identity_logit_model())
        x = jax.random.normal(jax.random.PRNGKey(0), (64, 3))
        y = jax.random.randint(jax.random.PRNGKey(1), (64,), 0, 3)
        out = wrapper.fit(x, y)
        assert out is wrapper
        logits = predict_raw(wrapper, jax.random.normal(jax.random.PRNGKey(2), (8, 3)))
        assert logits.shape == (8, 3)


class TestFlaxAffineLogitErrors:
    def test_temperature_property_returns_none_when_uncalibrated(self) -> None:
        """Uncalibrated wrappers return ``None`` for ``temperature`` and ``bias``."""
        _flax_modules()
        from probly.method.calibration import flax_identity_logit_model, temperature_scaling  # noqa: PLC0415

        wrapper = temperature_scaling(flax_identity_logit_model())
        assert wrapper.temperature is None
        assert wrapper.bias is None

    def test_bias_property_returns_none_when_use_bias_false(self) -> None:
        """``bias`` is None for methods that don't use a bias term, even when calibrated."""
        jax, _ = _flax_modules()
        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import flax_identity_logit_model, temperature_scaling  # noqa: PLC0415

        wrapper = temperature_scaling(flax_identity_logit_model())
        x = jax.random.normal(jax.random.PRNGKey(0), (64, 3))
        y = jax.random.randint(jax.random.PRNGKey(1), (64,), 0, 3)
        calibrate(wrapper, y, x)
        assert wrapper.is_calibrated is True
        assert wrapper.bias is None

    def test_calibrate_rejects_zero_dim_logits(self) -> None:
        """Logits with no dimensions cannot be calibrated."""
        _, jnp = _flax_modules()
        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import flax_identity_logit_model, temperature_scaling  # noqa: PLC0415

        wrapper = temperature_scaling(flax_identity_logit_model())
        with pytest.raises(ValueError, match="at least one dimension"):
            calibrate(wrapper, jnp.array(0), jnp.array(0.0))

    def test_calibrate_rejects_nonjax_predictor_output(self) -> None:
        """The flax wrapper requires the underlying predictor to return jax arrays."""
        _, jnp = _flax_modules()
        from flax import nnx  # noqa: PLC0415

        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import temperature_scaling  # noqa: PLC0415

        class NumpyOutputModel(nnx.Module):
            def __call__(self, x: object) -> object:
                import numpy as np  # noqa: PLC0415

                return np.asarray(x)

        wrapper = temperature_scaling(NumpyOutputModel())
        with pytest.raises(TypeError, match="Flax calibration expects jax logits"):
            calibrate(wrapper, jnp.zeros(4), jnp.zeros((4, 3)))

    def test_call_rejects_nonjax_predictor_output(self) -> None:
        """``__call__`` similarly requires the predictor to return jax arrays."""
        jax, _ = _flax_modules()
        from flax import nnx  # noqa: PLC0415

        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import temperature_scaling  # noqa: PLC0415
        from probly.predictor import predict_raw  # noqa: PLC0415

        class FlakyModel(nnx.Module):
            def __init__(self) -> None:
                super().__init__()
                self._calls = 0

            def __call__(self, x: object) -> object:
                self._calls += 1
                if self._calls <= 1:
                    return x
                import numpy as np  # noqa: PLC0415

                return np.asarray(x)

        wrapper = temperature_scaling(FlakyModel())
        x = jax.random.normal(jax.random.PRNGKey(0), (64, 3))
        y = jax.random.randint(jax.random.PRNGKey(1), (64,), 0, 3)
        calibrate(wrapper, y, x)
        with pytest.raises(TypeError, match="Flax calibration expects jax logits"):
            predict_raw(wrapper, jax.random.normal(jax.random.PRNGKey(2), (8, 3)))

    def test_validate_vector_logits_rejects_singleton_class_dim(self) -> None:
        """Vector scaling rejects logits without an explicit class axis."""
        jax, _ = _flax_modules()
        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import flax_identity_logit_model, vector_scaling  # noqa: PLC0415

        wrapper = vector_scaling(flax_identity_logit_model(), num_classes=3)
        x = jax.random.normal(jax.random.PRNGKey(0), (8,))
        y = jax.random.randint(jax.random.PRNGKey(1), (8,), 0, 3)
        with pytest.raises(ValueError, match="explicit class axis"):
            calibrate(wrapper, y, x)

    def test_validate_vector_logits_rejects_label_count_mismatch(self) -> None:
        """Labels must match the number of logit rows."""
        jax, _ = _flax_modules()
        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import flax_identity_logit_model, vector_scaling  # noqa: PLC0415

        wrapper = vector_scaling(flax_identity_logit_model(), num_classes=3)
        x = jax.random.normal(jax.random.PRNGKey(0), (8, 3))
        y = jax.random.randint(jax.random.PRNGKey(1), (5,), 0, 3)
        with pytest.raises(ValueError, match="must match logits batch size"):
            calibrate(wrapper, y, x)

    def test_validate_vector_logits_rejects_num_classes_mismatch(self) -> None:
        """Configured ``num_classes`` must match the data."""
        jax, _ = _flax_modules()
        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import flax_identity_logit_model, vector_scaling  # noqa: PLC0415

        wrapper = vector_scaling(flax_identity_logit_model(), num_classes=4)
        x = jax.random.normal(jax.random.PRNGKey(0), (8, 3))
        y = jax.random.randint(jax.random.PRNGKey(1), (8,), 0, 3)
        with pytest.raises(ValueError, match="Expected logits with 4 classes"):
            calibrate(wrapper, y, x)


class TestFlaxIsotonicErrors:
    def test_calibrate_rejects_nonjax_predictor_output(self) -> None:
        """Isotonic calibration requires the predictor to return jax arrays."""
        _, jnp = _flax_modules()
        from flax import nnx  # noqa: PLC0415

        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import isotonic_regression  # noqa: PLC0415
        from probly.predictor import BinaryLogitClassifier  # noqa: PLC0415

        class NumpyOutputModel(nnx.Module):
            def __call__(self, x: object) -> object:
                import numpy as np  # noqa: PLC0415

                return np.asarray(x)

        wrapper = isotonic_regression(NumpyOutputModel(), predictor_type=BinaryLogitClassifier)
        with pytest.raises(TypeError, match="jax predictions"):
            calibrate(wrapper, jnp.zeros(4), jnp.zeros(4))

    def test_call_rejects_nonjax_predictor_output(self) -> None:
        """The isotonic ``__call__`` enforces jax-array-typed predictor output."""
        jax, jnp = _flax_modules()
        from flax import nnx  # noqa: PLC0415

        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import isotonic_regression  # noqa: PLC0415
        from probly.predictor import BinaryLogitClassifier, predict_raw  # noqa: PLC0415

        class FlakyModel(nnx.Module):
            def __init__(self) -> None:
                super().__init__()
                self._calls = 0

            def __call__(self, x: object) -> object:
                self._calls += 1
                if self._calls <= 1:
                    return x
                import numpy as np  # noqa: PLC0415

                return np.asarray(x)

        wrapper = isotonic_regression(FlakyModel(), predictor_type=BinaryLogitClassifier)
        x = jax.random.normal(jax.random.PRNGKey(0), (64,))
        y = (x > 0).astype(jnp.float32)
        calibrate(wrapper, y, x)
        with pytest.raises(TypeError, match="jax logits"):
            predict_raw(wrapper, jax.random.normal(jax.random.PRNGKey(1), (8,)))

    def test_store_isotonic_knots_rejects_too_many_knots(self) -> None:
        """Storing more knots than the fixed nnx state layout raises a clear error."""
        _, jnp = _flax_modules()
        from probly.method.calibration import flax_identity_logit_model, isotonic_regression  # noqa: PLC0415
        from probly.predictor import BinaryLogitClassifier  # noqa: PLC0415
        from probly.transformation.calibration.flax import _ISOTONIC_MAX_KNOTS  # noqa: PLC0415

        wrapper = isotonic_regression(flax_identity_logit_model(), predictor_type=BinaryLogitClassifier)
        too_many_x = jnp.zeros(_ISOTONIC_MAX_KNOTS + 1)
        too_many_y = jnp.zeros(_ISOTONIC_MAX_KNOTS + 1)
        with pytest.raises(ValueError, match="more knots than supported"):
            wrapper._store_isotonic_knots(too_many_x, too_many_y)  # noqa: SLF001

    def test_require_isotonic_knots_rejects_zero_knots(self) -> None:
        """An isotonic wrapper with the calibrated flag flipped manually but no knots raises."""
        _, jnp = _flax_modules()
        from probly.method.calibration import flax_identity_logit_model, isotonic_regression  # noqa: PLC0415
        from probly.predictor import BinaryLogitClassifier  # noqa: PLC0415

        wrapper = isotonic_regression(flax_identity_logit_model(), predictor_type=BinaryLogitClassifier)
        wrapper._is_calibrated = jnp.asarray(True)  # noqa: SLF001
        wrapper._isotonic_num_knots = jnp.asarray(0)  # noqa: SLF001
        with pytest.raises(ValueError, match="no fitted knots"):
            wrapper._require_isotonic_knots()  # noqa: SLF001

    def test_require_isotonic_knots_rejects_uncalibrated(self) -> None:
        """An uncalibrated isotonic wrapper rejects calls to ``_require_isotonic_knots``."""
        _flax_modules()
        from probly.method.calibration import flax_identity_logit_model, isotonic_regression  # noqa: PLC0415
        from probly.predictor import BinaryLogitClassifier  # noqa: PLC0415

        wrapper = isotonic_regression(flax_identity_logit_model(), predictor_type=BinaryLogitClassifier)
        with pytest.raises(ValueError, match="not calibrated"):
            wrapper._require_isotonic_knots()  # noqa: SLF001

    def test_apply_isotonic_single_knot_branch(self) -> None:
        """When isotonic regression collapses to a single knot, predictions broadcast that constant."""
        _, jnp = _flax_modules()
        from probly.method.calibration import flax_identity_logit_model, isotonic_regression  # noqa: PLC0415
        from probly.predictor import BinaryLogitClassifier, predict_raw  # noqa: PLC0415

        wrapper = isotonic_regression(flax_identity_logit_model(), predictor_type=BinaryLogitClassifier)
        # Simulate the rare degenerate case where there is only one knot.
        wrapper._isotonic_x_knots = wrapper._isotonic_x_knots.at[0].set(0.0)  # noqa: SLF001
        wrapper._isotonic_y_knots = wrapper._isotonic_y_knots.at[0].set(0.4)  # noqa: SLF001
        wrapper._isotonic_num_knots = jnp.asarray(1)  # noqa: SLF001
        wrapper._is_calibrated = jnp.asarray(True)  # noqa: SLF001
        out = predict_raw(wrapper, jnp.array([0.5, -1.0, 2.0]))
        assert jnp.allclose(out, jnp.full_like(out, 0.4))


class TestFlaxDirichletCalibration:
    NUM_CLASSES = 3
    NUM_SAMPLES = 512
    SHARPENING = 3.0

    @classmethod
    def _overconfident_logits(cls, num_classes: int | None = None) -> tuple[jax_types.Array, jax_types.Array]:
        """Create synthetic overconfident logits and labels for calibration tests."""
        jax, jnp = _flax_modules()
        num_classes = cls.NUM_CLASSES if num_classes is None else num_classes
        label_key, logit_key = jax.random.split(jax.random.PRNGKey(0))
        labels = jax.random.randint(label_key, (cls.NUM_SAMPLES,), 0, num_classes)
        base = jax.random.normal(logit_key, (cls.NUM_SAMPLES, num_classes))
        # Push probability mass toward the true class, then sharpen to overconfidence.
        base = base.at[jnp.arange(cls.NUM_SAMPLES), labels].add(1.5)
        return base * cls.SHARPENING, labels

    @staticmethod
    def _nll(logits: jax_types.Array, labels: jax_types.Array) -> float:
        jax, jnp = _flax_modules()
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        picked = jnp.take_along_axis(log_probs, labels.astype(jnp.int32)[:, None], axis=-1)
        return float(-jnp.mean(picked))

    def test_call_shape_and_calibrate_returns_logits(self) -> None:
        """Calibrated output keeps the class axis and a finite range."""
        _, jnp = _flax_modules()
        from probly.method.calibration import dirichlet_calibration, flax_identity_logit_model  # noqa: PLC0415
        from probly.predictor import predict_raw  # noqa: PLC0415

        logits, labels = self._overconfident_logits()
        model = dirichlet_calibration(flax_identity_logit_model(), num_classes=self.NUM_CLASSES)
        assert model.calibrate(labels, logits) is model
        out = predict_raw(model, logits)
        assert out.shape == logits.shape
        assert bool(jnp.isfinite(out).all())

    def test_calibration_reduces_nll_for_five_classes(self) -> None:
        """The fit converges to a substantial NLL improvement beyond the three-class case."""
        _flax_modules()
        from probly.method.calibration import dirichlet_calibration, flax_identity_logit_model  # noqa: PLC0415
        from probly.predictor import predict_raw  # noqa: PLC0415

        logits, labels = self._overconfident_logits(num_classes=5)
        model = dirichlet_calibration(flax_identity_logit_model(), num_classes=5)
        model.calibrate(labels, logits)
        calibrated = predict_raw(model, logits)
        assert self._nll(calibrated, labels) < 0.9 * self._nll(logits, labels)

    def test_strong_off_diagonal_regularisation_shrinks_off_diagonal(self) -> None:
        """A large reg_lambda drives the off-diagonal weights toward zero."""
        _, jnp = _flax_modules()
        from probly.method.calibration import dirichlet_calibration, flax_identity_logit_model  # noqa: PLC0415

        logits, labels = self._overconfident_logits()
        weak = dirichlet_calibration(
            flax_identity_logit_model(), num_classes=self.NUM_CLASSES, reg_lambda=0.0, reg_mu=0.0
        )
        strong = dirichlet_calibration(
            flax_identity_logit_model(), num_classes=self.NUM_CLASSES, reg_lambda=1e3, reg_mu=0.0
        )
        weak.calibrate(labels, logits)
        strong.calibrate(labels, logits)

        eye = jnp.eye(self.NUM_CLASSES, dtype=bool)
        weak_off = jnp.abs(weak.weight[~eye]).mean()
        strong_off = jnp.abs(strong.weight[~eye]).mean()
        assert float(strong_off) < float(weak_off)

    def test_strong_intercept_regularisation_shrinks_bias(self) -> None:
        """A large reg_mu drives the fitted bias toward zero."""
        _, jnp = _flax_modules()
        from probly.method.calibration import dirichlet_calibration, flax_identity_logit_model  # noqa: PLC0415

        logits, labels = self._overconfident_logits()
        weak = dirichlet_calibration(flax_identity_logit_model(), num_classes=self.NUM_CLASSES, reg_mu=0.0)
        strong = dirichlet_calibration(flax_identity_logit_model(), num_classes=self.NUM_CLASSES, reg_mu=1e3)
        weak.calibrate(labels, logits)
        strong.calibrate(labels, logits)

        assert float(jnp.abs(strong.bias).mean()) < float(jnp.abs(weak.bias).mean())

    def test_generic_calibrate_matches_fit_alias(self) -> None:
        """The generic calibrate() and the sklearn-style fit() alias agree."""
        _, jnp = _flax_modules()
        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import dirichlet_calibration, flax_identity_logit_model  # noqa: PLC0415

        logits, labels = self._overconfident_logits()
        via_calibrate = dirichlet_calibration(flax_identity_logit_model(), num_classes=self.NUM_CLASSES)
        calibrate(via_calibrate, labels, logits)

        via_fit = dirichlet_calibration(flax_identity_logit_model(), num_classes=self.NUM_CLASSES)
        via_fit.fit(logits, labels)

        assert bool(jnp.allclose(via_calibrate.weight, via_fit.weight, atol=1e-5))
        assert bool(jnp.allclose(via_calibrate.bias, via_fit.bias, atol=1e-5))

    def test_reg_mu_defaults_to_reg_lambda(self) -> None:
        """When reg_mu is None it inherits reg_lambda."""
        _flax_modules()
        from probly.method.calibration import dirichlet_calibration, flax_identity_logit_model  # noqa: PLC0415

        model = dirichlet_calibration(flax_identity_logit_model(), num_classes=self.NUM_CLASSES, reg_lambda=0.25)
        assert model.reg_mu == pytest.approx(0.25)
        assert model.reg_lambda == pytest.approx(0.25)

    @pytest.mark.parametrize("num_classes", [None, 1, 0])
    def test_invalid_num_classes_raises(self, num_classes: int | None) -> None:
        """num_classes must be greater than one."""
        _flax_modules()
        from probly.method.calibration import dirichlet_calibration, flax_identity_logit_model  # noqa: PLC0415

        with pytest.raises(ValueError, match="num_classes"):
            dirichlet_calibration(flax_identity_logit_model(), num_classes=num_classes)

    @pytest.mark.parametrize("num_classes", [None, 1])
    def test_wrapper_rejects_invalid_num_classes_config(self, num_classes: int | None) -> None:
        """Constructing the wrapper directly also validates the configured class count."""
        _flax_modules()
        from probly.method.calibration import flax_identity_logit_model  # noqa: PLC0415
        from probly.transformation.calibration._common import CalibrationMethodConfig  # noqa: PLC0415
        from probly.transformation.calibration.flax import FlaxDirichletCalibrationPredictor  # noqa: PLC0415

        config = CalibrationMethodConfig(method="dirichlet", use_bias=True, num_classes=num_classes)
        with pytest.raises(ValueError, match="num_classes"):
            FlaxDirichletCalibrationPredictor(flax_identity_logit_model(), config)

    def test_weight_and_bias_are_none_when_uncalibrated(self) -> None:
        """Uncalibrated wrappers expose no fitted parameters."""
        _flax_modules()
        from probly.method.calibration import dirichlet_calibration, flax_identity_logit_model  # noqa: PLC0415

        model = dirichlet_calibration(flax_identity_logit_model(), num_classes=self.NUM_CLASSES)
        assert model.is_calibrated is False
        assert model.weight is None
        assert model.bias is None

    def test_fitted_parameter_shapes(self) -> None:
        """The fitted map is a full ``k x k`` matrix with a length ``k`` bias."""
        _flax_modules()
        from probly.method.calibration import dirichlet_calibration, flax_identity_logit_model  # noqa: PLC0415

        logits, labels = self._overconfident_logits()
        model = dirichlet_calibration(flax_identity_logit_model(), num_classes=self.NUM_CLASSES)
        model.calibrate(labels, logits)
        assert model.is_calibrated is True
        assert model.weight.shape == (self.NUM_CLASSES, self.NUM_CLASSES)
        assert model.bias.shape == (self.NUM_CLASSES,)

    def test_predict_before_calibrate_raises(self) -> None:
        """Prediction before calibration is rejected."""
        _flax_modules()
        from probly.method.calibration import dirichlet_calibration, flax_identity_logit_model  # noqa: PLC0415
        from probly.predictor import predict_raw  # noqa: PLC0415

        logits, _ = self._overconfident_logits()
        model = dirichlet_calibration(flax_identity_logit_model(), num_classes=self.NUM_CLASSES)
        with pytest.raises(ValueError, match="not calibrated"):
            predict_raw(model, logits)

    def test_state_from_mismatched_num_classes_raises(self) -> None:
        """State restored from a wrapper with a different class count is rejected at prediction."""
        _, jnp = _flax_modules()
        from flax import nnx  # noqa: PLC0415

        from probly.method.calibration import dirichlet_calibration, flax_identity_logit_model  # noqa: PLC0415
        from probly.predictor import predict_raw  # noqa: PLC0415

        logits, labels = self._overconfident_logits()
        fitted = dirichlet_calibration(flax_identity_logit_model(), num_classes=self.NUM_CLASSES)
        fitted.calibrate(labels, logits)

        other = dirichlet_calibration(flax_identity_logit_model(), num_classes=self.NUM_CLASSES + 1)
        nnx.update(other, nnx.state(fitted))
        with pytest.raises(ValueError, match="do not match"):
            predict_raw(other, jnp.zeros((2, self.NUM_CLASSES + 1)))

    def test_mismatched_class_axis_raises(self) -> None:
        """Logits whose class axis disagrees with num_classes are rejected."""
        _flax_modules()
        from probly.method.calibration import dirichlet_calibration, flax_identity_logit_model  # noqa: PLC0415

        logits, labels = self._overconfident_logits()
        model = dirichlet_calibration(flax_identity_logit_model(), num_classes=self.NUM_CLASSES + 1)
        with pytest.raises(ValueError, match="class axis"):
            model.calibrate(labels, logits)

    def test_logits_without_class_axis_raise(self) -> None:
        """Dirichlet calibration needs an explicit class axis, so 1-D logits are rejected."""
        _, jnp = _flax_modules()
        from probly.method.calibration import dirichlet_calibration, flax_identity_logit_model  # noqa: PLC0415

        model = dirichlet_calibration(flax_identity_logit_model(), num_classes=self.NUM_CLASSES)
        with pytest.raises(ValueError, match="class axis"):
            model.calibrate(jnp.zeros(4, dtype=jnp.int32), jnp.zeros(4))

    def test_label_count_mismatch_raises(self) -> None:
        """Labels must match the number of logit rows."""
        _flax_modules()
        from probly.method.calibration import dirichlet_calibration, flax_identity_logit_model  # noqa: PLC0415

        logits, labels = self._overconfident_logits()
        model = dirichlet_calibration(flax_identity_logit_model(), num_classes=self.NUM_CLASSES)
        with pytest.raises(ValueError, match="must match logits batch size"):
            model.calibrate(labels[:-1], logits)

    def test_out_of_range_labels_raise(self) -> None:
        """Labels outside [0, num_classes) are rejected instead of silently clamped."""
        _flax_modules()
        from probly.method.calibration import dirichlet_calibration, flax_identity_logit_model  # noqa: PLC0415

        logits, labels = self._overconfident_logits()
        model = dirichlet_calibration(flax_identity_logit_model(), num_classes=self.NUM_CLASSES)
        with pytest.raises(ValueError, match="must lie in"):
            model.calibrate(labels.at[0].set(self.NUM_CLASSES), logits)
        with pytest.raises(ValueError, match="must lie in"):
            model.calibrate(labels.at[0].set(-1), logits)

    def test_empty_calibration_set_raises(self) -> None:
        """Calibrating on zero samples is rejected instead of fitting a silent no-op."""
        _, jnp = _flax_modules()
        from probly.method.calibration import dirichlet_calibration, flax_identity_logit_model  # noqa: PLC0415

        model = dirichlet_calibration(flax_identity_logit_model(), num_classes=self.NUM_CLASSES)
        with pytest.raises(ValueError, match="non-empty"):
            model.calibrate(jnp.zeros((0,), jnp.int32), jnp.zeros((0, self.NUM_CLASSES)))

    def test_non_finite_logits_raise(self) -> None:
        """Non-finite calibration logits are rejected instead of poisoning the fitted state."""
        _, jnp = _flax_modules()
        from probly.method.calibration import dirichlet_calibration, flax_identity_logit_model  # noqa: PLC0415

        logits, labels = self._overconfident_logits()
        model = dirichlet_calibration(flax_identity_logit_model(), num_classes=self.NUM_CLASSES)
        with pytest.raises(ValueError, match="non-finite"):
            model.calibrate(labels, logits.at[0, 0].set(-jnp.inf))

    def test_accepts_non_jax_labels(self) -> None:
        """Labels given as a plain sequence are converted before fitting."""
        _flax_modules()
        from probly.method.calibration import dirichlet_calibration, flax_identity_logit_model  # noqa: PLC0415

        logits, labels = self._overconfident_logits()
        model = dirichlet_calibration(flax_identity_logit_model(), num_classes=self.NUM_CLASSES)
        model.calibrate([int(label) for label in labels], logits)
        assert model.is_calibrated is True

    def test_call_preserves_leading_batch_dimensions(self) -> None:
        """Prediction broadcasts the calibration map over arbitrary leading dimensions."""
        _, jnp = _flax_modules()
        from probly.method.calibration import dirichlet_calibration, flax_identity_logit_model  # noqa: PLC0415
        from probly.predictor import predict_raw  # noqa: PLC0415

        logits, labels = self._overconfident_logits()
        model = dirichlet_calibration(flax_identity_logit_model(), num_classes=self.NUM_CLASSES)
        model.calibrate(labels, logits)

        batched = logits.reshape(4, -1, self.NUM_CLASSES)
        out = predict_raw(model, batched)
        assert out.shape == batched.shape
        assert bool(jnp.allclose(out.reshape(-1, self.NUM_CLASSES), predict_raw(model, logits), atol=1e-6))

    def test_calibrate_rejects_nonjax_predictor_output(self) -> None:
        """Dirichlet calibration requires the predictor to return jax arrays."""
        _flax_modules()
        from flax import nnx  # noqa: PLC0415

        from probly.method.calibration import dirichlet_calibration  # noqa: PLC0415

        class NumpyOutputModel(nnx.Module):
            def __call__(self, x: object) -> object:
                import numpy as np  # noqa: PLC0415

                return np.asarray(x)

        logits, labels = self._overconfident_logits()
        model = dirichlet_calibration(NumpyOutputModel(), num_classes=self.NUM_CLASSES)
        with pytest.raises(TypeError, match="Flax Dirichlet calibration expects jax logits"):
            model.calibrate(labels, logits)
