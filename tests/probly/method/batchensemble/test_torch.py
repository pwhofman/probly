"""Test for torch batchensemble models."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from probly.layers.torch import BatchEnsembleConv2d, BatchEnsembleLinear  # noqa: E402
from probly.method.batchensemble import batchensemble  # noqa: E402
from probly.predictor import predict  # noqa: E402
from probly.representation.sample.torch import TorchSample  # noqa: E402
from probly.representer import DummyRepresenter, representer  # noqa: E402
from tests.probly.torch_utils import count_layers  # noqa: E402


class TestBatchEnsembleRepresenter:
    """Tests for batchensemble representer registration."""

    def test_batchensemble_uses_dummy_representer(self) -> None:
        model = batchensemble(nn.Linear(4, 2), num_members=3)
        rep = representer(model)
        assert isinstance(rep, DummyRepresenter)

    def test_batchensemble_representer_returns_torch_sample(self) -> None:
        model = batchensemble(nn.Linear(4, 2), num_members=3)
        rep = representer(model)
        out = rep.represent(torch.ones(2, 4))
        assert isinstance(out, TorchSample)
        assert out.tensor.shape == (3, 2, 2)


class TestBatchEnsembleLayers:
    """Tests for BatchEnsembleLinear and BatchEnsembleConv2d layer generation."""

    @pytest.mark.parametrize(
        "model_fixture",
        [
            "torch_model_small_2d_2d",
            "torch_conv_linear_model",
            "torch_regression_model_1d",
            "torch_regression_model_2d",
            "torch_dropout_model",
        ],
    )
    def test_batchensemble_generation(
        self,
        request: pytest.FixtureRequest,
        model_fixture: str,
    ) -> None:
        model = request.getfixturevalue(model_fixture)
        num_members = 5

        batchensemble_model = batchensemble(model, num_members)
        # ``batchensemble`` traverses the module in place and tags it with ``num_members``;
        # the model itself preserves the original structure (e.g. nn.Sequential) and takes
        # ``[E*B, ...]`` inputs. The user-facing predict() handler tiles for you.
        assert isinstance(batchensemble_model, nn.Sequential)
        assert batchensemble_model.num_members == num_members

        count_linear_original = count_layers(model, nn.Linear)
        count_convolutional_original = count_layers(model, nn.Conv2d)
        count_sequential_original = count_layers(model, nn.Sequential)

        count_linear_modified = count_layers(batchensemble_model, nn.Linear)
        count_convolutional_modified = count_layers(batchensemble_model, nn.Conv2d)
        count_sequential_modified = count_layers(batchensemble_model, nn.Sequential)

        count_batchensemblelinear_modified = count_layers(batchensemble_model, BatchEnsembleLinear)
        count_batchensembleconv2d_modified = count_layers(batchensemble_model, BatchEnsembleConv2d)

        assert count_batchensemblelinear_modified == count_linear_original
        assert count_batchensembleconv2d_modified == count_convolutional_original
        assert count_linear_modified == 0
        assert count_convolutional_modified == 0
        assert count_sequential_original == count_sequential_modified

    @pytest.mark.parametrize(
        "model_fixture",
        [
            "torch_model_small_2d_2d",
            "torch_conv_linear_model",
            "torch_regression_model_1d",
            "torch_regression_model_2d",
            "torch_dropout_model",
        ],
    )
    def test_batchensemble_shapes(
        self,
        request: pytest.FixtureRequest,
        model_fixture: str,
    ) -> None:
        model = request.getfixturevalue(model_fixture)
        num_members = 5
        batchensemble_model = batchensemble(model, num_members)

        # The traversed model has the same children layout as the input; only Linear/Conv
        # layers are swapped for their BatchEnsemble counterparts.
        for fixture_layer, batchensemble_layer in zip(model.children(), batchensemble_model.children(), strict=False):
            if isinstance(fixture_layer, nn.Linear) and isinstance(batchensemble_layer, BatchEnsembleLinear):
                fixture_in_features = fixture_layer.in_features
                fixture_out_features = fixture_layer.out_features

                batchensemble_in_features = batchensemble_layer.in_features
                batchensemble_out_features = batchensemble_layer.out_features
                batchensemble_layer_s = batchensemble_layer.s.shape
                batchensemble_layer_r = batchensemble_layer.r.shape

                # assert if in_features and out_features are the same
                assert batchensemble_in_features == fixture_in_features
                assert batchensemble_out_features == fixture_out_features

                # assert if the shapes are correct (r modulates input dim, s modulates output dim)
                assert batchensemble_layer_r == (num_members, batchensemble_in_features)
                assert batchensemble_layer_s == (num_members, batchensemble_out_features)
                assert batchensemble_layer.bias.shape == (num_members, batchensemble_out_features)

            if isinstance(fixture_layer, nn.Conv2d) and isinstance(batchensemble_layer, BatchEnsembleConv2d):
                fixture_in_channels = fixture_layer.in_channels
                fixture_out_channels = fixture_layer.out_channels

                batchensemble_in_channels = batchensemble_layer.in_channels
                batchensemble_out_channels = batchensemble_layer.out_channels
                batchensemble_layer_s = batchensemble_layer.s.shape
                batchensemble_layer_r = batchensemble_layer.r.shape

                # assert if in_channels and out_channels are the same
                assert batchensemble_in_channels == fixture_in_channels
                assert batchensemble_out_channels == fixture_out_channels

                # assert if the shapes are correct (r modulates input channels, s modulates output channels)
                assert batchensemble_layer_r == (num_members, batchensemble_in_channels)
                assert batchensemble_layer_s == (num_members, batchensemble_out_channels)
                assert batchensemble_layer.bias.shape == (num_members, batchensemble_out_channels)

    def test_custom_model(self, torch_custom_model) -> None:
        num_members = 5
        model = batchensemble(torch_custom_model, num_members)
        # Type-preserving: the traversed model is still the same class as the input.
        assert isinstance(model, type(torch_custom_model))
        assert not isinstance(model, nn.Sequential)
        assert model.num_members == num_members

        count_linear_original = count_layers(torch_custom_model, nn.Linear)
        count_convolutional_original = count_layers(torch_custom_model, nn.Conv2d)
        count_sequential_original = count_layers(torch_custom_model, nn.Sequential)

        count_linear_modified = count_layers(model, nn.Linear)
        count_convolutional_modified = count_layers(model, nn.Conv2d)
        count_sequential_modified = count_layers(model, nn.Sequential)

        count_batchensemblelinear_modified = count_layers(model, BatchEnsembleLinear)
        count_batchensembleconv2d_modified = count_layers(model, BatchEnsembleConv2d)

        assert count_linear_modified == 0
        assert count_convolutional_modified == 0
        assert count_batchensemblelinear_modified == count_linear_original
        assert count_batchensembleconv2d_modified == count_convolutional_original
        assert count_sequential_modified == count_sequential_original

    def test_batchensemble_prints(self, capsys) -> None:
        in_features = 1
        out_features = 2
        kernel_size = 2
        stride = 2
        num_members = 3
        batchensemble_linear = batchensemble(nn.Linear(in_features, out_features), num_members=num_members)

        # Single Linear in -> single BatchEnsembleLinear out (no wrapper).
        assert isinstance(batchensemble_linear, BatchEnsembleLinear)
        print(batchensemble_linear)  # noqa: T201
        captured = capsys.readouterr()
        linear_output = captured.out.strip()

        expected_linear = (
            f"BatchEnsembleLinear(in_features={in_features}, out_features={out_features},"
            f" num_members={num_members}, bias=True)"
        )

        assert linear_output == expected_linear

        batchensemble_conv2d = batchensemble(
            nn.Conv2d(in_features, out_features, kernel_size, stride=stride), num_members=num_members
        )

        assert isinstance(batchensemble_conv2d, BatchEnsembleConv2d)
        print(batchensemble_conv2d)  # noqa: T201
        captured = capsys.readouterr()
        conv2d_output = captured.out.strip()

        expected_conv2d = (
            f"BatchEnsembleConv2d(in_channels={in_features}, out_channels={out_features},"
            f" kernel_size={(kernel_size, kernel_size)}, stride={(stride, stride)}, num_members={num_members})"
        )

        assert conv2d_output == expected_conv2d


class TestBatchEnsembleForwards:
    """Test class for BatchEnsemble layer forwards."""

    def test_batchensemble_layer_forwards(self) -> None:
        """Tests forwards of BatchEnsembleLinear and BatchEnsembleConv2d.

        The model takes ``[E*B, ...]`` inputs and returns ``[E*B, ...]`` outputs (the
        "pure" forward signature, matching edward2). The user-facing ``predict()`` handler
        wraps these into a :class:`TorchSample` with ``sample_dim=0``.
        """
        batch_size = 2
        out_dim = 2
        kernel_size = 1
        x_linear = torch.ones(batch_size, out_dim)
        x_conv2d = torch.ones(batch_size, out_dim, kernel_size, kernel_size)

        num_members = 5
        batchensemble_linear = batchensemble(
            nn.Linear(out_dim, out_dim), num_members=num_members, use_base_weights=True
        )
        batchensemble_conv2d = batchensemble(
            nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size), num_members=num_members, use_base_weights=True
        )

        # Direct call: pre-tile the input by num_members.
        be_linear_out = batchensemble_linear(x_linear.repeat(num_members, 1))
        be_conv2d_out = batchensemble_conv2d(x_conv2d.repeat(num_members, 1, 1, 1))
        assert be_linear_out.shape == (num_members * batch_size, out_dim)
        assert be_conv2d_out.shape == (num_members * batch_size, out_dim, kernel_size, kernel_size)

        # predict(): tiles internally and wraps in TorchSample with sample_dim=0.
        linear_sample = predict(batchensemble_linear, x_linear)
        conv2d_sample = predict(batchensemble_conv2d, x_conv2d)
        assert isinstance(linear_sample, TorchSample)
        assert isinstance(conv2d_sample, TorchSample)
        assert linear_sample.sample_dim == 0
        assert conv2d_sample.sample_dim == 0
        assert linear_sample.tensor.shape == (num_members, batch_size, out_dim)
        assert conv2d_sample.tensor.shape == (num_members, batch_size, out_dim, kernel_size, kernel_size)

    def test_batchensemble_torch_custom_model_forward(self, torch_custom_model) -> None:
        """Tests forward of transformed torch_custom_model via ``predict``."""
        num_members = 5
        batchensemble_model = batchensemble(
            torch_custom_model,
            num_members=num_members,
            use_base_weights=True,
        )

        batch_size = 2
        in_dim = 10
        out_dim = 4

        x = torch.ones(batch_size, in_dim)
        sample = predict(batchensemble_model, x)
        assert isinstance(sample, TorchSample)
        assert sample.sample_dim == 0
        assert sample.tensor.shape == (num_members, batch_size, out_dim)

    def test_batchensemble_forward_errors(self) -> None:
        """The model expects ``[E*B, ...]``; mis-shaped inputs raise from the layer."""
        linear = nn.Linear(1, 2)
        conv2d = nn.Conv2d(1, 2, 1)

        num_members = 2
        batchensemble_linear = batchensemble(linear, num_members=num_members)
        batchensemble_conv2d = batchensemble(conv2d, num_members=num_members)

        # 3D input to BatchEnsembleLinear.
        with pytest.raises(ValueError, match=r"Expected 2D input \[E\*B, in_features\]"):
            batchensemble_linear(torch.ones(2, 1, 2))

        # 5D input to BatchEnsembleConv2d.
        with pytest.raises(ValueError, match=r"Expected 4D input \[E\*B, C, H, W\]"):
            batchensemble_conv2d(torch.ones(2, 1, 1, 1, 2))

        # Batch size not divisible by num_members.
        with pytest.raises(ValueError, match=r"is not divisible by num_members"):
            batchensemble_linear(torch.ones(3, 1))
