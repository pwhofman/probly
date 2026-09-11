"""Test for torch bayesian models."""

from __future__ import annotations

from typing import cast

import pytest

from probly.layers.torch import BayesConv2d, BayesLinear
from probly.method.bayesian import bayesian
from tests.probly.torch_utils import count_layers

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, IterableDataset, TensorDataset  # noqa: E402

from probly.method.bayesian.torch import train_bayesian  # noqa: E402


class TestNetworkArchitectures:
    """Test class for different network architectures."""

    def test_linear_network_replacement(
        self,
        torch_model_small_2d_2d: nn.Sequential,
    ) -> None:
        """Tests if a model incorporates a bayesian layer correctly when a linear layer is present.

        This function verifies that:
        - A standard linear layer is replaced with a bayesian linear layer.
        """
        model = cast("nn.Module", bayesian(torch_model_small_2d_2d))

        # count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_model_small_2d_2d, nn.Linear)
        # count number of BayesLinear layers in original model
        count_bayesian_original = count_layers(torch_model_small_2d_2d, BayesLinear)
        # count number of nn.Sequential layers in original model
        count_sequential_original = count_layers(torch_model_small_2d_2d, nn.Sequential)

        # count number of nn.Linear layers in modified model
        count_linear_modified = count_layers(model, nn.Linear)
        # count number of BayesLinear layers in modified model
        count_bayesian_modified = count_layers(model, BayesLinear)
        # count number of nn.Sequential layers in modified model
        count_sequential_modified = count_layers(model, nn.Sequential)

        # check that the model is not modified except for the bayesian layer
        assert model is not None
        assert isinstance(model, type(torch_model_small_2d_2d))
        assert count_bayesian_modified == count_bayesian_original + count_linear_original
        assert count_linear_modified == 0
        assert count_sequential_original == count_sequential_modified

    def test_convolutional_network(self, torch_conv_linear_model: nn.Sequential) -> None:
        """Tests the convolutional neural network modification with added bayesian layers.

        This function evaluates whether the given convolutional neural network model
        has been correctly modified to include bayesian layers without altering the
        number of other components such as linear, sequential, or convolutional layers.
        """
        model = cast("nn.Module", bayesian(torch_conv_linear_model))

        # count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_conv_linear_model, nn.Linear)
        # count number of BayesConv2d layers in original model
        count_bayesian_conv_original = count_layers(torch_conv_linear_model, BayesConv2d)
        # count number of nn.Sequential layers in original model
        count_sequential_original = count_layers(torch_conv_linear_model, nn.Sequential)
        # count number of nn.Conv2d layers in original model
        count_conv_original = count_layers(torch_conv_linear_model, nn.Conv2d)
        # count number of BayesLinear layers in original model
        count_bayesian_linear_original = count_layers(torch_conv_linear_model, BayesLinear)

        # count number of nn.Linear layers in modified model
        count_linear_modified = count_layers(model, nn.Linear)
        # count number of BayesConv2d layers in modified model
        count_bayesian_conv_modified = count_layers(model, BayesConv2d)
        # count number of nn.Sequential layers in modified model
        count_sequential_modified = count_layers(model, nn.Sequential)
        # count number of nn.Conv2d layers in modified model
        count_conv_modified = count_layers(model, nn.Conv2d)
        # count number of BayesLinear layers in modified model
        count_bayesian_linear_modified = count_layers(model, BayesLinear)

        # check that the model is not modified except for the bayesian layer
        assert model is not None
        assert isinstance(model, type(torch_conv_linear_model))
        assert count_linear_modified == 0
        assert count_conv_modified == 0
        assert count_bayesian_conv_modified == count_bayesian_conv_original + count_conv_original
        assert count_bayesian_linear_modified == count_bayesian_linear_original + count_linear_original
        assert count_sequential_original == count_sequential_modified

    def test_custom_network(self, torch_custom_model: nn.Module) -> None:
        """Tests the custom model modification with added bayesian layers."""
        model = bayesian(torch_custom_model)

        # check if model type is correct
        assert isinstance(model, type(torch_custom_model))
        assert not isinstance(model, nn.Sequential)


def _separable_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    inputs = torch.cat([torch.randn(24, 2) - 3.0, torch.randn(24, 2) + 3.0])
    targets = torch.arange(2).repeat_interleave(24)
    return inputs, targets


def _separable_loader() -> DataLoader:
    inputs, targets = _separable_data()
    return DataLoader(TensorDataset(inputs, targets), batch_size=16, shuffle=False)


class _StreamDataset(IterableDataset):
    """Length-less dataset: yields the separable data as a stream."""

    def __iter__(self):  # noqa: ANN204
        inputs, targets = _separable_data()
        yield from zip(inputs, targets, strict=True)


def _bayesian_predictor():
    torch.manual_seed(1)
    base = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
    return bayesian(base, predictor_type="logit_classifier")


class TestTrainBayesian:
    """ELBO training of a single Bayesian predictor."""

    def test_elbo_training_reduces_loss(self) -> None:
        records: list[dict[str, float]] = []
        train_bayesian(_bayesian_predictor(), _separable_loader(), kl_scale=0.1, epochs=5, on_epoch=records.append)
        assert all(set(r) == {"epoch", "running_loss"} for r in records)
        assert records[-1]["running_loss"] < records[0]["running_loss"]

    def test_kl_term_scales_the_objective(self) -> None:
        losses: dict[float, float] = {}
        for kl_scale in (0.0, 1e4):
            records: list[dict[str, float]] = []
            train_bayesian(
                _bayesian_predictor(), _separable_loader(), kl_scale=kl_scale, epochs=1, on_epoch=records.append
            )
            losses[kl_scale] = records[0]["running_loss"]
        assert losses[1e4] > losses[0.0] + 1.0

    def test_val_loss_reported(self) -> None:
        records: list[dict[str, float]] = []
        train_bayesian(
            _bayesian_predictor(),
            _separable_loader(),
            val_loader=_separable_loader(),
            epochs=1,
            on_epoch=records.append,
        )
        assert "val_loss" in records[0]

    def test_length_less_dataset_requires_dataset_size(self) -> None:
        stream_loader = DataLoader(_StreamDataset(), batch_size=16)
        with pytest.raises(TypeError, match="dataset_size"):
            train_bayesian(_bayesian_predictor(), stream_loader)
        train_bayesian(_bayesian_predictor(), stream_loader, dataset_size=48, epochs=1)

    def test_nonpositive_dataset_size_raises(self) -> None:
        with pytest.raises(ValueError, match="dataset_size"):
            train_bayesian(_bayesian_predictor(), _separable_loader(), dataset_size=0)
