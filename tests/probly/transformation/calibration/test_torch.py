"""Tests for the small helper functions in ``transformation.calibration.torch``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch as torch_types


def _torch_modules():
    pytest.importorskip("torch")
    import torch  # noqa: PLC0415

    return torch


class TestInverseSoftplus:
    def test_inverse_relation(self) -> None:
        torch = _torch_modules()
        from probly.transformation.calibration.torch import _inverse_softplus  # noqa: PLC0415

        x = torch.tensor([0.5, 1.0, 2.0])
        y = _inverse_softplus(x)
        # softplus(y) = log(1 + exp(y)). _inverse_softplus(x) = log(expm1(x)).
        # Verify softplus(_inverse_softplus(x)) = x for x > 0.
        torch.testing.assert_close(torch.nn.functional.softplus(y), x, atol=1e-5, rtol=1e-5)


class TestReshapeBinary:
    def test_squeezes_trailing_singleton(self) -> None:
        torch = _torch_modules()
        from probly.transformation.calibration.torch import _reshape_binary_preds  # noqa: PLC0415

        preds = torch.zeros((3, 1))
        out = _reshape_binary_preds(preds)
        assert out.shape == (3,)

    def test_keeps_higher_dim(self) -> None:
        torch = _torch_modules()
        from probly.transformation.calibration.torch import _reshape_binary_preds  # noqa: PLC0415

        preds = torch.zeros((3,))
        out = _reshape_binary_preds(preds)
        # 1D unchanged.
        assert out.shape == (3,)


class TestReshapeBinaryLabels:
    def test_match(self) -> None:
        torch = _torch_modules()
        from probly.transformation.calibration.torch import _reshape_binary_labels  # noqa: PLC0415

        labels = torch.tensor([[0, 1], [1, 0]])
        out = _reshape_binary_labels(labels, expected_elements=4)
        assert out.shape == (4,)

    def test_mismatch_raises(self) -> None:
        torch = _torch_modules()
        from probly.transformation.calibration.torch import _reshape_binary_labels  # noqa: PLC0415

        with pytest.raises(ValueError, match="Binary calibration labels"):
            _reshape_binary_labels(torch.tensor([0, 1]), expected_elements=3)


class TestReshapeMulticlassLabels:
    def test_match(self) -> None:
        torch = _torch_modules()
        from probly.transformation.calibration.torch import _reshape_multiclass_labels  # noqa: PLC0415

        labels = torch.tensor([[0, 1], [1, 0]])
        out = _reshape_multiclass_labels(labels, batch_shape=(2, 2))
        assert out.shape == (4,)

    def test_mismatch_raises(self) -> None:
        torch = _torch_modules()
        from probly.transformation.calibration.torch import _reshape_multiclass_labels  # noqa: PLC0415

        with pytest.raises(ValueError, match="Multiclass calibration labels"):
            _reshape_multiclass_labels(torch.tensor([0]), batch_shape=(2,))

    def test_empty_batch_shape(self) -> None:
        torch = _torch_modules()
        from probly.transformation.calibration.torch import _reshape_multiclass_labels  # noqa: PLC0415

        # Batch shape () -> expected 1 element.
        out = _reshape_multiclass_labels(torch.tensor([3]), batch_shape=())
        assert out.shape == (1,)


class TestCalibrationLoss:
    def test_multiclass_loss(self) -> None:
        torch = _torch_modules()
        from probly.transformation.calibration.torch import _calibration_loss  # noqa: PLC0415

        scaled_logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        labels = torch.tensor([2, 0])
        loss = _calibration_loss(scaled_logits, labels)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_binary_loss(self) -> None:
        torch = _torch_modules()
        from probly.transformation.calibration.torch import _calibration_loss  # noqa: PLC0415

        scaled_logits = torch.tensor([0.5, -0.5, 1.0])
        labels = torch.tensor([1, 0, 1])
        loss = _calibration_loss(scaled_logits, labels)
        assert torch.isfinite(loss)


class TestPrepareBinaryIsotonicInputs:
    def test_returns_flat_inputs(self) -> None:
        torch = _torch_modules()
        from probly.transformation.calibration.torch import _prepare_binary_isotonic_inputs  # noqa: PLC0415

        preds = torch.zeros(5)
        labels = torch.zeros(5)
        flat_logits, flat_labels, singleton = _prepare_binary_isotonic_inputs(preds, labels)
        assert flat_logits.shape == (5,)
        assert flat_labels.shape == (5,)
        assert singleton is False

    def test_singleton_class_axis(self) -> None:
        torch = _torch_modules()
        from probly.transformation.calibration.torch import _prepare_binary_isotonic_inputs  # noqa: PLC0415

        preds = torch.zeros((5, 1))
        labels = torch.zeros(5)
        flat_logits, flat_labels, singleton = _prepare_binary_isotonic_inputs(preds, labels)  # noqa: RUF059
        assert singleton is True

    def test_zero_dim_raises(self) -> None:
        torch = _torch_modules()
        from probly.transformation.calibration.torch import _prepare_binary_isotonic_inputs  # noqa: PLC0415

        with pytest.raises(ValueError, match="at least one dimension"):
            _prepare_binary_isotonic_inputs(torch.tensor(0.5), torch.tensor(0.0))

    def test_multiclass_raises(self) -> None:
        torch = _torch_modules()
        from probly.transformation.calibration.torch import _prepare_binary_isotonic_inputs  # noqa: PLC0415

        with pytest.raises(ValueError, match="binary logits only"):
            _prepare_binary_isotonic_inputs(torch.zeros((5, 3)), torch.zeros(5))

    def test_shape_mismatch_raises(self) -> None:
        torch = _torch_modules()
        from probly.transformation.calibration.torch import _prepare_binary_isotonic_inputs  # noqa: PLC0415

        with pytest.raises(ValueError, match="batch size"):
            _prepare_binary_isotonic_inputs(torch.zeros(5), torch.zeros(3))


class TestApplyAffine:
    def test_scalar_temperature(self) -> None:
        torch = _torch_modules()
        from probly.transformation.calibration.torch import _apply_affine  # noqa: PLC0415

        logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        T = torch.tensor([2.0])  # noqa: N806
        out = _apply_affine(logits, T, bias=None)
        torch.testing.assert_close(out, logits / 2.0)

    def test_per_class_temperature(self) -> None:
        torch = _torch_modules()
        from probly.transformation.calibration.torch import _apply_affine  # noqa: PLC0415

        logits = torch.tensor([[1.0, 4.0]])
        T = torch.tensor([1.0, 2.0])  # noqa: N806
        out = _apply_affine(logits, T, bias=None)
        torch.testing.assert_close(out, torch.tensor([[1.0, 2.0]]))

    def test_with_scalar_bias(self) -> None:
        torch = _torch_modules()
        from probly.transformation.calibration.torch import _apply_affine  # noqa: PLC0415

        logits = torch.tensor([[1.0, 2.0]])
        T = torch.tensor([1.0])  # noqa: N806
        bias = torch.tensor([0.5])
        out = _apply_affine(logits, T, bias=bias)
        torch.testing.assert_close(out, torch.tensor([[1.5, 2.5]]))

    def test_with_per_class_bias(self) -> None:
        torch = _torch_modules()
        from probly.transformation.calibration.torch import _apply_affine  # noqa: PLC0415

        logits = torch.tensor([[1.0, 2.0]])
        T = torch.tensor([1.0, 1.0])  # noqa: N806
        bias = torch.tensor([0.1, 0.2])
        out = _apply_affine(logits, T, bias=bias)
        torch.testing.assert_close(out, torch.tensor([[1.1, 2.2]]))

    def test_zero_dim_raises(self) -> None:
        torch = _torch_modules()
        from probly.transformation.calibration.torch import _apply_affine  # noqa: PLC0415

        with pytest.raises(ValueError, match="at least one dimension"):
            _apply_affine(torch.tensor(0.5), torch.tensor([1.0]), None)

    def test_temperature_shape_mismatch_raises(self) -> None:
        torch = _torch_modules()
        from probly.transformation.calibration.torch import _apply_affine  # noqa: PLC0415

        logits = torch.tensor([[1.0, 2.0]])
        T = torch.tensor([1.0, 1.0, 1.0])  # wrong size  # noqa: N806
        with pytest.raises(ValueError, match="Temperature"):
            _apply_affine(logits, T, None)

    def test_bias_shape_mismatch_raises(self) -> None:
        torch = _torch_modules()
        from probly.transformation.calibration.torch import _apply_affine  # noqa: PLC0415

        logits = torch.tensor([[1.0, 2.0]])
        T = torch.tensor([1.0, 1.0])  # noqa: N806
        bias = torch.tensor([0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="Bias"):
            _apply_affine(logits, T, bias)


class TestTorchIdentityLogitModel:
    def test_forward_returns_unchanged(self) -> None:
        torch = _torch_modules()
        from probly.transformation.calibration.torch import TorchIdentityLogitModel  # noqa: PLC0415

        m = TorchIdentityLogitModel()
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = m(x)
        torch.testing.assert_close(out, x)


class TestTorchCalibrationPredictorBaseHelpers:
    def test_abstract_calibrate_raises(self) -> None:
        """Subclasses must implement ``calibrate``."""
        torch = _torch_modules()
        from probly.transformation.calibration._common import CalibrationMethodConfig  # noqa: PLC0415
        from probly.transformation.calibration.torch import _TorchCalibrationPredictorBase  # noqa: PLC0415

        config = CalibrationMethodConfig(method="temperature", vector_scale=False, use_bias=False)

        class MinimalSubclass(_TorchCalibrationPredictorBase):
            def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - shape stub
                return x

            # Intentionally call ``super().calibrate`` to hit the NotImplementedError branch.
            def calibrate(self, y_calib: torch.Tensor, *args: object, **kwargs: object) -> object:
                return _TorchCalibrationPredictorBase.calibrate(self, y_calib, *args, **kwargs)

        instance = MinimalSubclass(torch.nn.Identity(), config)
        with pytest.raises(NotImplementedError):
            instance.calibrate(torch.zeros(3))

    def test_fit_swaps_argument_order(self) -> None:
        """``fit(x, y)`` reorders args to call ``calibrate(y, x)``."""
        torch = _torch_modules()
        from probly.method.calibration import temperature_scaling, torch_identity_logit_model  # noqa: PLC0415
        from probly.predictor import predict_raw  # noqa: PLC0415

        wrapper = temperature_scaling(torch_identity_logit_model())
        x = torch.randn(64, 3)
        y = torch.randint(0, 3, (64,))
        out = wrapper.fit(x, y)
        assert out is wrapper
        logits = predict_raw(wrapper, torch.randn(8, 3))
        assert logits.shape == (8, 3)


class TestTorchAffineLogitErrors:
    def test_temperature_property_returns_none_when_uncalibrated(self) -> None:
        """Uncalibrated wrappers return ``None`` for ``temperature`` and ``bias``."""
        _torch_modules()
        from probly.method.calibration import temperature_scaling, torch_identity_logit_model  # noqa: PLC0415

        wrapper = temperature_scaling(torch_identity_logit_model())
        assert wrapper.temperature is None
        assert wrapper.bias is None

    def test_bias_property_returns_none_when_use_bias_false(self) -> None:
        """``bias`` is None for methods that don't use a bias term, even when calibrated."""
        torch = _torch_modules()
        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import temperature_scaling, torch_identity_logit_model  # noqa: PLC0415

        wrapper = temperature_scaling(torch_identity_logit_model())
        x = torch.randn(64, 3)
        y = torch.randint(0, 3, (64,))
        calibrate(wrapper, y, x)
        assert wrapper.is_calibrated is True
        assert wrapper.bias is None

    def test_calibrate_rejects_zero_dim_logits(self) -> None:
        """Logits with no dimensions cannot be calibrated."""
        torch = _torch_modules()
        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import temperature_scaling, torch_identity_logit_model  # noqa: PLC0415

        wrapper = temperature_scaling(torch_identity_logit_model())
        with pytest.raises(ValueError, match="at least one dimension"):
            calibrate(wrapper, torch.tensor(0), torch.tensor(0.0))

    def test_calibrate_rejects_nontorch_predictor_output(self) -> None:
        """The torch wrapper requires the underlying predictor to return tensors."""
        torch = _torch_modules()
        from torch import nn  # noqa: PLC0415

        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import temperature_scaling  # noqa: PLC0415

        class NumpyOutputModel(nn.Module):
            def forward(self, x: torch.Tensor) -> object:
                return x.cpu().numpy()  # not a torch tensor

        wrapper = temperature_scaling(NumpyOutputModel())
        with pytest.raises(TypeError, match="Torch calibration expects torch logits"):
            calibrate(wrapper, torch.zeros(4), torch.zeros(4, 3))

    def test_forward_rejects_nontorch_predictor_output(self) -> None:
        """``forward`` similarly requires the predictor to return tensors."""
        torch = _torch_modules()
        from torch import nn  # noqa: PLC0415

        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import temperature_scaling  # noqa: PLC0415
        from probly.predictor import predict_raw  # noqa: PLC0415

        # Create a wrapper that predicts tensors at calibration time but switches behaviour
        # on the second call.
        class FlakyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._calls = 0

            def forward(self, x: torch.Tensor) -> object:
                self._calls += 1
                if self._calls <= 1:
                    return x
                return x.cpu().numpy()

        wrapper = temperature_scaling(FlakyModel())
        x = torch.randn(64, 3)
        y = torch.randint(0, 3, (64,))
        calibrate(wrapper, y, x)
        with pytest.raises(TypeError, match="Torch calibration expects torch logits"):
            predict_raw(wrapper, torch.randn(8, 3))

    def test_validate_vector_logits_rejects_singleton_class_dim(self) -> None:
        """Vector scaling rejects logits without an explicit class axis."""
        torch = _torch_modules()
        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import torch_identity_logit_model, vector_scaling  # noqa: PLC0415

        wrapper = vector_scaling(torch_identity_logit_model(), num_classes=3)
        x = torch.randn(8)  # 1-D — no class axis
        y = torch.randint(0, 3, (8,))
        with pytest.raises(ValueError, match="explicit class axis"):
            calibrate(wrapper, y, x)

    def test_validate_vector_logits_rejects_label_count_mismatch(self) -> None:
        """Labels must match the number of logit rows."""
        torch = _torch_modules()
        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import torch_identity_logit_model, vector_scaling  # noqa: PLC0415

        wrapper = vector_scaling(torch_identity_logit_model(), num_classes=3)
        x = torch.randn(8, 3)
        y = torch.randint(0, 3, (5,))  # mismatched
        with pytest.raises(ValueError, match="must match logits batch size"):
            calibrate(wrapper, y, x)

    def test_validate_vector_logits_rejects_num_classes_mismatch(self) -> None:
        """Configured ``num_classes`` must match the data."""
        torch = _torch_modules()
        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import torch_identity_logit_model, vector_scaling  # noqa: PLC0415

        wrapper = vector_scaling(torch_identity_logit_model(), num_classes=4)
        x = torch.randn(8, 3)
        y = torch.randint(0, 3, (8,))
        with pytest.raises(ValueError, match="Expected logits with 4 classes"):
            calibrate(wrapper, y, x)


class TestTorchAffineLogitMissingBuffers:
    def test_temperature_buffer_missing_raises_on_predict(self) -> None:
        """If the temperature buffer is somehow missing, prediction raises."""
        torch = _torch_modules()
        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import temperature_scaling, torch_identity_logit_model  # noqa: PLC0415
        from probly.predictor import predict_raw  # noqa: PLC0415

        wrapper = temperature_scaling(torch_identity_logit_model())
        x = torch.randn(64, 3)
        y = torch.randint(0, 3, (64,))
        calibrate(wrapper, y, x)
        # Force the temperature buffer to a non-tensor.
        wrapper._buffers["_temperature"] = None  # noqa: SLF001
        with pytest.raises(ValueError, match="Calibrated temperature buffer"):
            predict_raw(wrapper, x)

    def test_bias_buffer_missing_raises_on_predict(self) -> None:
        """If the bias buffer is missing for a use_bias method, prediction raises."""
        torch = _torch_modules()
        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import platt_scaling, torch_identity_logit_model  # noqa: PLC0415
        from probly.predictor import predict_raw  # noqa: PLC0415

        wrapper = platt_scaling(torch_identity_logit_model())
        x = torch.randn(64)
        y = torch.randint(0, 2, (64,)).float()
        calibrate(wrapper, y, x)
        wrapper._buffers["_bias"] = None  # noqa: SLF001
        with pytest.raises(ValueError, match="Calibrated bias buffer"):
            predict_raw(wrapper, x)

    def test_temperature_property_falls_back_to_none_on_bad_buffer(self) -> None:
        """The ``temperature`` property returns ``None`` if the buffer is invalid."""
        torch = _torch_modules()
        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import temperature_scaling, torch_identity_logit_model  # noqa: PLC0415

        wrapper = temperature_scaling(torch_identity_logit_model())
        x = torch.randn(64, 3)
        y = torch.randint(0, 3, (64,))
        calibrate(wrapper, y, x)
        wrapper._buffers["_temperature"] = None  # noqa: SLF001
        assert wrapper.temperature is None

    def test_bias_property_falls_back_to_none_on_bad_buffer(self) -> None:
        """The ``bias`` property returns ``None`` if the buffer is invalid."""
        torch = _torch_modules()
        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import platt_scaling, torch_identity_logit_model  # noqa: PLC0415

        wrapper = platt_scaling(torch_identity_logit_model())
        x = torch.randn(64)
        y = torch.randint(0, 2, (64,)).float()
        calibrate(wrapper, y, x)
        wrapper._buffers["_bias"] = None  # noqa: SLF001
        assert wrapper.bias is None


class TestTorchIsotonicErrors:
    def test_calibrate_rejects_nontorch_predictor_output(self) -> None:
        """Isotonic calibration requires the predictor to return tensors."""
        torch = _torch_modules()
        from torch import nn  # noqa: PLC0415

        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import isotonic_regression  # noqa: PLC0415
        from probly.predictor import BinaryLogitClassifier  # noqa: PLC0415

        class NumpyOutputModel(nn.Module):
            def forward(self, x: torch.Tensor) -> object:
                return x.cpu().numpy()

        wrapper = isotonic_regression(NumpyOutputModel(), predictor_type=BinaryLogitClassifier)
        with pytest.raises(TypeError, match="torch predictions"):
            calibrate(wrapper, torch.zeros(4), torch.zeros(4))

    def test_forward_rejects_nontorch_predictor_output(self) -> None:
        """The isotonic forward method enforces tensor-typed predictor output."""
        torch = _torch_modules()
        from torch import nn  # noqa: PLC0415

        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import isotonic_regression  # noqa: PLC0415
        from probly.predictor import BinaryLogitClassifier, predict_raw  # noqa: PLC0415

        class FlakyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._calls = 0

            def forward(self, x: torch.Tensor) -> object:
                self._calls += 1
                if self._calls <= 1:
                    return x
                return x.cpu().numpy()

        wrapper = isotonic_regression(FlakyModel(), predictor_type=BinaryLogitClassifier)
        x = torch.randn(64)
        y = (x > 0).float()
        calibrate(wrapper, y, x)
        with pytest.raises(TypeError, match="torch logits"):
            predict_raw(wrapper, torch.randn(8))

    def test_store_isotonic_knots_rejects_too_many_knots(self) -> None:
        """Storing more knots than the fixed buffer length raises a clear error."""
        torch = _torch_modules()
        from probly.method.calibration import isotonic_regression, torch_identity_logit_model  # noqa: PLC0415
        from probly.predictor import BinaryLogitClassifier  # noqa: PLC0415
        from probly.transformation.calibration.torch import _ISOTONIC_MAX_KNOTS  # noqa: PLC0415

        wrapper = isotonic_regression(torch_identity_logit_model(), predictor_type=BinaryLogitClassifier)
        too_many_x = torch.zeros(_ISOTONIC_MAX_KNOTS + 1)
        too_many_y = torch.zeros(_ISOTONIC_MAX_KNOTS + 1)
        with pytest.raises(ValueError, match="more knots than supported"):
            wrapper._store_isotonic_knots(too_many_x, too_many_y)  # noqa: SLF001

    def test_store_isotonic_knots_rejects_missing_buffers(self) -> None:
        """Missing isotonic buffers raise a clear ``TypeError``."""
        torch = _torch_modules()
        from probly.method.calibration import isotonic_regression, torch_identity_logit_model  # noqa: PLC0415
        from probly.predictor import BinaryLogitClassifier  # noqa: PLC0415

        wrapper = isotonic_regression(torch_identity_logit_model(), predictor_type=BinaryLogitClassifier)
        wrapper._buffers["_isotonic_x_knots"] = None  # noqa: SLF001
        with pytest.raises(TypeError, match="Isotonic calibration buffers"):
            wrapper._store_isotonic_knots(torch.zeros(2), torch.zeros(2))  # noqa: SLF001

    def test_require_isotonic_knots_rejects_bad_buffers(self) -> None:
        """Invalid isotonic buffers raise on prediction."""
        torch = _torch_modules()
        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import isotonic_regression, torch_identity_logit_model  # noqa: PLC0415
        from probly.predictor import BinaryLogitClassifier  # noqa: PLC0415

        wrapper = isotonic_regression(torch_identity_logit_model(), predictor_type=BinaryLogitClassifier)
        x = torch.randn(64)
        y = (x > 0).float()
        calibrate(wrapper, y, x)
        wrapper._buffers["_isotonic_num_knots"] = None  # noqa: SLF001
        with pytest.raises(TypeError, match="Isotonic calibration buffers"):
            wrapper._require_isotonic_knots()  # noqa: SLF001

    def test_require_isotonic_knots_rejects_zero_knots(self) -> None:
        """An isotonic wrapper with the calibrated flag flipped manually but no knots raises."""
        _torch_modules()
        from probly.method.calibration import isotonic_regression, torch_identity_logit_model  # noqa: PLC0415
        from probly.predictor import BinaryLogitClassifier  # noqa: PLC0415

        wrapper = isotonic_regression(torch_identity_logit_model(), predictor_type=BinaryLogitClassifier)
        # Manually flip the calibration flag without storing knots.
        wrapper._buffers["_is_calibrated"].fill_(True)  # noqa: SLF001
        wrapper._buffers["_isotonic_num_knots"].fill_(0)  # noqa: SLF001
        with pytest.raises(ValueError, match="no fitted knots"):
            wrapper._require_isotonic_knots()  # noqa: SLF001

    def test_require_isotonic_knots_rejects_uncalibrated(self) -> None:
        """An uncalibrated isotonic wrapper rejects calls to ``_require_isotonic_knots``."""
        _torch_modules()
        from probly.method.calibration import isotonic_regression, torch_identity_logit_model  # noqa: PLC0415
        from probly.predictor import BinaryLogitClassifier  # noqa: PLC0415

        wrapper = isotonic_regression(torch_identity_logit_model(), predictor_type=BinaryLogitClassifier)
        with pytest.raises(ValueError, match="not calibrated"):
            wrapper._require_isotonic_knots()  # noqa: SLF001

    def test_apply_isotonic_single_knot_branch(self) -> None:
        """When isotonic regression collapses to a single knot, predictions broadcast that constant."""
        torch = _torch_modules()
        from probly.method.calibration import isotonic_regression, torch_identity_logit_model  # noqa: PLC0415
        from probly.predictor import BinaryLogitClassifier, predict_raw  # noqa: PLC0415

        wrapper = isotonic_regression(torch_identity_logit_model(), predictor_type=BinaryLogitClassifier)
        # Simulate the rare degenerate case where there is only one knot.
        with torch.no_grad():
            wrapper._buffers["_isotonic_x_knots"][0] = 0.0  # noqa: SLF001
            wrapper._buffers["_isotonic_y_knots"][0] = 0.4  # noqa: SLF001
            wrapper._buffers["_isotonic_num_knots"].fill_(1)  # noqa: SLF001
            wrapper._buffers["_is_calibrated"].fill_(True)  # noqa: SLF001
        out = predict_raw(wrapper, torch.tensor([0.5, -1.0, 2.0]))
        torch.testing.assert_close(out, torch.full_like(out, 0.4))


class TestTorchDirichletCalibration:
    NUM_CLASSES = 3
    NUM_SAMPLES = 512
    SHARPENING = 3.0

    @classmethod
    def _overconfident_logits(cls, num_classes: int | None = None) -> tuple[torch_types.Tensor, torch_types.Tensor]:
        """Create synthetic overconfident logits and labels for calibration tests."""
        torch = _torch_modules()
        num_classes = cls.NUM_CLASSES if num_classes is None else num_classes
        torch.manual_seed(0)
        labels = torch.randint(0, num_classes, (cls.NUM_SAMPLES,))
        base = torch.randn(cls.NUM_SAMPLES, num_classes)
        # Push probability mass toward the true class, then sharpen to overconfidence.
        base[torch.arange(cls.NUM_SAMPLES), labels] += 1.5
        return base * cls.SHARPENING, labels

    @staticmethod
    def _nll(logits: torch_types.Tensor, labels: torch_types.Tensor) -> float:
        _torch_modules()
        import torch.nn.functional as F  # noqa: PLC0415

        return float(F.cross_entropy(logits, labels.long()).item())

    def test_forward_shape_and_calibrate_returns_logits(self) -> None:
        """Calibrated output keeps the class axis and a finite range."""
        torch = _torch_modules()
        from probly.method.calibration import dirichlet_calibration, torch_identity_logit_model  # noqa: PLC0415
        from probly.predictor import predict_raw  # noqa: PLC0415

        logits, labels = self._overconfident_logits()
        model = dirichlet_calibration(torch_identity_logit_model(), num_classes=self.NUM_CLASSES)
        model.calibrate(labels, logits)
        out = predict_raw(model, logits)
        assert out.shape == logits.shape
        assert torch.isfinite(out).all()

    def test_calibration_reduces_nll(self) -> None:
        """Fitting Dirichlet calibration lowers NLL on overconfident logits."""
        _torch_modules()
        from probly.method.calibration import dirichlet_calibration, torch_identity_logit_model  # noqa: PLC0415
        from probly.predictor import predict_raw  # noqa: PLC0415

        logits, labels = self._overconfident_logits()
        model = dirichlet_calibration(torch_identity_logit_model(), num_classes=self.NUM_CLASSES)
        model.calibrate(labels, logits)
        calibrated = predict_raw(model, logits)
        assert self._nll(calibrated, labels) < self._nll(logits, labels)

    def test_strong_off_diagonal_regularisation_shrinks_off_diagonal(self) -> None:
        """A large reg_lambda drives the off-diagonal weights toward zero."""
        torch = _torch_modules()
        from probly.method.calibration import dirichlet_calibration, torch_identity_logit_model  # noqa: PLC0415

        logits, labels = self._overconfident_logits()
        weak = dirichlet_calibration(torch_identity_logit_model(), num_classes=self.NUM_CLASSES, reg_lambda=0.0)
        strong = dirichlet_calibration(torch_identity_logit_model(), num_classes=self.NUM_CLASSES, reg_lambda=1e3)
        weak.calibrate(labels, logits)
        strong.calibrate(labels, logits)

        eye = torch.eye(self.NUM_CLASSES, dtype=torch.bool)
        weak_off = weak.weight[~eye].abs().mean()
        strong_off = strong.weight[~eye].abs().mean()
        assert strong_off < weak_off

    def test_generic_calibrate_matches_fit_alias(self) -> None:
        """The generic calibrate() and the sklearn-style fit() alias agree."""
        torch = _torch_modules()
        from probly.calibrator import calibrate  # noqa: PLC0415
        from probly.method.calibration import dirichlet_calibration, torch_identity_logit_model  # noqa: PLC0415

        logits, labels = self._overconfident_logits()
        via_calibrate = dirichlet_calibration(torch_identity_logit_model(), num_classes=self.NUM_CLASSES)
        calibrate(via_calibrate, labels, logits)

        via_fit = dirichlet_calibration(torch_identity_logit_model(), num_classes=self.NUM_CLASSES)
        via_fit.fit(logits, labels)

        assert torch.allclose(via_calibrate.weight, via_fit.weight, atol=1e-5)
        assert torch.allclose(via_calibrate.bias, via_fit.bias, atol=1e-5)

    def test_reg_mu_defaults_to_reg_lambda(self) -> None:
        """When reg_mu is None it inherits reg_lambda."""
        _torch_modules()
        from probly.method.calibration import dirichlet_calibration, torch_identity_logit_model  # noqa: PLC0415

        model = dirichlet_calibration(torch_identity_logit_model(), num_classes=self.NUM_CLASSES, reg_lambda=0.25)
        assert model.reg_mu == pytest.approx(0.25)

    @pytest.mark.parametrize("num_classes", [None, 1, 0])
    def test_invalid_num_classes_raises(self, num_classes: int | None) -> None:
        """num_classes must be greater than one."""
        _torch_modules()
        from probly.method.calibration import dirichlet_calibration, torch_identity_logit_model  # noqa: PLC0415

        with pytest.raises(ValueError, match="num_classes"):
            dirichlet_calibration(torch_identity_logit_model(), num_classes=num_classes)

    def test_predict_before_calibrate_raises(self) -> None:
        """Prediction before calibration is rejected."""
        _torch_modules()
        from probly.method.calibration import dirichlet_calibration, torch_identity_logit_model  # noqa: PLC0415
        from probly.predictor import predict_raw  # noqa: PLC0415

        logits, _ = self._overconfident_logits()
        model = dirichlet_calibration(torch_identity_logit_model(), num_classes=self.NUM_CLASSES)
        with pytest.raises(ValueError, match="not calibrated"):
            predict_raw(model, logits)

    def test_state_dict_round_trip_reproduces_predictions(self) -> None:
        """A reloaded calibrator reproduces the fitted predictions."""
        torch = _torch_modules()
        from probly.method.calibration import dirichlet_calibration, torch_identity_logit_model  # noqa: PLC0415
        from probly.predictor import predict_raw  # noqa: PLC0415

        logits, labels = self._overconfident_logits()
        model = dirichlet_calibration(torch_identity_logit_model(), num_classes=self.NUM_CLASSES)
        model.calibrate(labels, logits)
        expected = predict_raw(model, logits)

        restored = dirichlet_calibration(torch_identity_logit_model(), num_classes=self.NUM_CLASSES)
        restored.load_state_dict(model.state_dict())
        assert torch.allclose(predict_raw(restored, logits), expected, atol=1e-6)

    def test_exposed_from_method_calibration_namespace(self) -> None:
        """dirichlet_calibration is re-exported through probly.method.calibration."""
        from probly.method.calibration import dirichlet_calibration as from_method  # noqa: PLC0415
        from probly.transformation.calibration import dirichlet_calibration as from_transformation  # noqa: PLC0415

        assert from_method is from_transformation

    def test_mismatched_class_axis_raises(self) -> None:
        """Logits whose class axis disagrees with num_classes are rejected."""
        _torch_modules()
        from probly.method.calibration import dirichlet_calibration, torch_identity_logit_model  # noqa: PLC0415

        logits, labels = self._overconfident_logits()
        model = dirichlet_calibration(torch_identity_logit_model(), num_classes=self.NUM_CLASSES + 1)
        with pytest.raises(ValueError, match="class axis"):
            model.calibrate(labels, logits)

    def test_forward_preserves_leading_batch_dimensions(self) -> None:
        """Prediction broadcasts the calibration map over arbitrary leading dimensions."""
        torch = _torch_modules()
        from probly.method.calibration import dirichlet_calibration, torch_identity_logit_model  # noqa: PLC0415
        from probly.predictor import predict_raw  # noqa: PLC0415

        logits, labels = self._overconfident_logits()
        model = dirichlet_calibration(torch_identity_logit_model(), num_classes=self.NUM_CLASSES)
        model.calibrate(labels, logits)

        batched = logits.reshape(4, -1, self.NUM_CLASSES)
        out = predict_raw(model, batched)
        assert out.shape == batched.shape
        assert torch.allclose(out.reshape(-1, self.NUM_CLASSES), predict_raw(model, logits), atol=1e-6)
