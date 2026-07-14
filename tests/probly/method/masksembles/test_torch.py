"""Torch-specific tests for masksembles."""

from __future__ import annotations

import re

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from probly.layers.torch import Masksembles2D, MasksemblesLinear  # noqa: E402
from probly.predictor import predict  # noqa: E402
from probly.quantification import quantify  # noqa: E402
from probly.quantification.notion import AleatoricUncertainty, EpistemicUncertainty, TotalUncertainty  # noqa: E402
from probly.representation.distribution.torch_categorical import (  # noqa: E402
    TorchCategoricalDistributionSample,
    TorchLogitCategoricalDistribution,
)
from probly.representation.sample.torch import TorchSample  # noqa: E402
from probly.representer import representer  # noqa: E402
from probly.transformation.masksembles import masksembles  # noqa: E402
from probly.transformation.masksembles._common import (  # noqa: E402
    MasksemblesPredictor,
    MasksemblesRepresenter,
    _wrap_masksembles_logits,
)
from probly.transformation.masksembles.torch import generation_wrapper, tile_inputs  # noqa: E402
from tests.probly.torch_utils import count_layers  # noqa: E402


@pytest.fixture
def linear_model() -> nn.Module:
    """MLP with features >= 10 throughout (required by generation_wrapper)."""
    return nn.Sequential(
        nn.Linear(20, 16),
        nn.ReLU(),
        nn.Linear(16, 10),
    )


@pytest.fixture
def conv_model() -> nn.Module:
    """Small CNN with channels >= 10 (required by generation_wrapper)."""
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * 8 * 8, 10),
    )


N_MASKS = 4
SCALE = 2.0


class TestGenerationWrapper:
    """Tests for the mask generation utility."""

    def test_raises_when_features_below_minimum(self) -> None:
        with pytest.raises(ValueError, match="less than 10"):
            generation_wrapper(c=9, n=4, scale=2.0)

    def test_raises_when_scale_over_maximum(self) -> None:
        match_msg = "scale parameter must be less than 6.0"
        with pytest.raises(ValueError, match=re.escape(match_msg)):
            generation_wrapper(c=10, n=4, scale=7.0)

    def test_raises_when_scale_negative(self) -> None:
        match_msg = "scale parameter must be less than 6.0 and positive"
        with pytest.raises(ValueError, match=re.escape(match_msg)):
            generation_wrapper(c=10, n=4, scale=-1.0)

    def test_raises_when_n_negative(self) -> None:
        with pytest.raises(ValueError, match="when number of masks is negative"):
            generation_wrapper(c=10, n=-1, scale=2.0)

    def test_output_shape(self) -> None:

        masks = generation_wrapper(c=16, n=4, scale=2.0)
        assert masks.shape == (4, 16)

    def test_raises_when_no_fitting_masks(self) -> None:
        with pytest.raises(ValueError, match="generation_wrapper was unable to generate"):
            generation_wrapper(c=10, n=2, scale=2.0)

    def test_masks_are_binary(self) -> None:

        masks = generation_wrapper(c=16, n=4, scale=2.0)
        unique = masks.unique().tolist()
        assert set(unique).issubset({0.0, 1.0})


class TestTileInputs:
    """Tests for the tile_inputs helper."""

    def test_tile_inputs_batch_shape(self) -> None:
        x = torch.randn(4, 20)
        out = tile_inputs(x, N_MASKS)
        assert out.shape == (N_MASKS * 4, 20)

    def test_tile_inputs_preserves_channels_last(self) -> None:
        x = torch.randn(4, 3, 8, 8).contiguous(memory_format=torch.channels_last)
        out = tile_inputs(x, N_MASKS)
        assert out.is_contiguous(memory_format=torch.channels_last)


class TestMasksemblesRepresenter:
    """Tests for MasksemblesRepresenter dispatch and output types."""

    def test_representer_dispatches_to_masksembles_representer(self, linear_model) -> None:
        model = masksembles(linear_model, n_masks=N_MASKS, scale=SCALE, predictor_type="logit_classifier")
        rep = representer(model)
        assert isinstance(rep, MasksemblesRepresenter)

    def test_represent_returns_categorical_distribution_sample(self, linear_model) -> None:
        model = masksembles(linear_model, n_masks=N_MASKS, scale=SCALE, predictor_type="logit_classifier")
        rep = representer(model)
        with torch.no_grad():
            out = rep.represent(torch.randn(5, 20))
        assert isinstance(out, TorchCategoricalDistributionSample)
        assert isinstance(out.tensor, TorchLogitCategoricalDistribution)

    def test_represent_output_shape(self, linear_model) -> None:
        # Output tensor: [B, n_masks, num_classes]; sample_dim=1 (mask axis).
        model = masksembles(linear_model, n_masks=N_MASKS, scale=SCALE, predictor_type="logit_classifier")
        rep = representer(model)
        with torch.no_grad():
            out = rep.represent(torch.randn(5, 20))
        assert out.tensor.tensor.shape == (5, N_MASKS, 10)
        assert out.sample_dim == 1

    def test_quantify_yields_per_sample_scalar_uncertainty(self, linear_model) -> None:
        model = masksembles(linear_model, n_masks=N_MASKS, scale=SCALE, predictor_type="logit_classifier")
        rep = representer(model)
        with torch.no_grad():
            out = rep.represent(torch.randn(7, 20))
        decomp = quantify(out)
        assert set(decomp.components) == {TotalUncertainty, AleatoricUncertainty, EpistemicUncertainty}
        assert decomp["total"].shape == (7,)


class TestMasksemblesForward:
    """Tests for MasksemblesLinearLayer and Masksembles2DLayer forward."""

    def test_linear_layer_train_mode_preserves_shape(self) -> None:
        layer = MasksemblesLinear(masks=generation_wrapper(16, N_MASKS, SCALE), features=16, n=N_MASKS, scale=SCALE)
        layer.train()
        x = torch.randn(8, 16)
        out = layer(x)
        assert out.shape == x.shape

    def test_linear_layer_eval_mode_preserves_shape(self) -> None:
        layer = MasksemblesLinear(masks=generation_wrapper(16, N_MASKS, SCALE), features=16, n=N_MASKS, scale=SCALE)
        layer.eval()
        x = torch.randn(N_MASKS * 8, 16)  # pre-tiled by n_masks
        out = layer(x)
        assert out.shape == x.shape

    def test_predict_masksembles_output_shape(self, linear_model) -> None:
        model = masksembles(linear_model, n_masks=N_MASKS, scale=SCALE, predictor_type="logit_classifier")
        x = torch.randn(5, 20)
        with torch.no_grad():
            sample = predict(model, x)
        assert isinstance(sample, TorchSample)
        assert sample.tensor.shape == (N_MASKS, 5, 10)
        assert sample.sample_dim == 0

    def test_predict_masksembles_conv_output_shape(self, conv_model) -> None:
        model = masksembles(conv_model, n_masks=N_MASKS, scale=SCALE, predictor_type="logit_classifier")
        x = torch.randn(3, 3, 8, 8)
        with torch.no_grad():
            sample = predict(model, x)
        assert isinstance(sample, TorchSample)
        assert sample.tensor.shape == (N_MASKS, 3, 10)
        assert sample.sample_dim == 0

    def test_predict_return_to_train_mode(self, linear_model) -> None:
        model = masksembles(linear_model, n_masks=N_MASKS, scale=SCALE, predictor_type="logit_classifier")
        x = torch.randn(5, 20)
        model.train()
        with torch.no_grad():
            predict(model, x)
        assert model.training is True

    def test_predict_return_to_eval_mode(self, linear_model) -> None:
        model = masksembles(linear_model, n_masks=N_MASKS, scale=SCALE, predictor_type="logit_classifier")
        x = torch.randn(5, 20)
        model.eval()
        with torch.no_grad():
            predict(model, x)
        assert model.training is False


class TestMasksemblesLayerInsertion:
    """Tests that masksembles() inserts mask layers at the right positions."""

    def test_linear_layers_are_appended(self, linear_model) -> None:
        model = masksembles(linear_model, n_masks=N_MASKS, scale=SCALE, predictor_type="logit_classifier")
        n_linear_original = count_layers(linear_model, nn.Linear)
        n_masked = count_layers(model, MasksemblesLinear)
        assert n_masked == n_linear_original - 1
        assert count_layers(model, MasksemblesLinear) > 0

    def test_last_linear_is_not_masked(self, linear_model) -> None:
        model = masksembles(linear_model, n_masks=N_MASKS, scale=SCALE, predictor_type="logit_classifier")
        n_linear_original = count_layers(linear_model, nn.Linear)
        n_masked = count_layers(model, MasksemblesLinear)
        assert n_masked == n_linear_original - 1

    def test_conv_layers_are_appended(self, conv_model) -> None:
        model = masksembles(conv_model, n_masks=N_MASKS, scale=SCALE, predictor_type="logit_classifier")
        n_conv_original = count_layers(model, nn.Conv2d)
        n_masked = count_layers(model, Masksembles2D)
        assert count_layers(model, Masksembles2D) > 0
        assert n_conv_original == n_masked

    def test_n_masks_buffer_is_attached(self, linear_model) -> None:
        model = masksembles(linear_model, n_masks=N_MASKS, scale=SCALE, predictor_type="logit_classifier")
        assert hasattr(model, "n_masks")
        assert int(model.n_masks) == N_MASKS

    def test_result_is_masksembles_predictor(self, linear_model) -> None:
        model = masksembles(linear_model, n_masks=N_MASKS, scale=SCALE, predictor_type="logit_classifier")
        assert isinstance(model, MasksemblesPredictor)


class TestWrapMasksemblesLogits:
    """Tests for _wrap_masksembles_logits: the sample_dim=0 -> sample_dim=1 transpose."""

    def test_transposes_sample_dim_from_0_to_1(self) -> None:
        # predict_masksembles returns sample_dim=0, shape [N, B, C].
        raw = TorchSample(tensor=torch.randn(N_MASKS, 5, 10), sample_dim=0)
        out = _wrap_masksembles_logits(raw)
        # After the transpose: shape [B, N, C], sample_dim=1.
        assert isinstance(out, TorchCategoricalDistributionSample)
        assert out.sample_dim == 1
        assert out.tensor.tensor.shape == (5, N_MASKS, 10)

    def test_no_transpose_when_sample_dim_is_not_0(self) -> None:
        # If sample_dim is already 1, no transpose should occur.
        raw = TorchSample(tensor=torch.randn(5, N_MASKS, 10), sample_dim=1)
        out = _wrap_masksembles_logits(raw)
        assert isinstance(out, TorchCategoricalDistributionSample)
        assert out.sample_dim == 1
        assert out.tensor.tensor.shape == (5, N_MASKS, 10)
