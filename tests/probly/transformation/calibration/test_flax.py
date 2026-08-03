"""Tests for the small helper functions in ``transformation.calibration.flax``."""

from __future__ import annotations

import pytest


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
