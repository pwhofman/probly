"""Tests for torch DropConnect transformation."""

from __future__ import annotations

import pytest
from tests.probly.torch_utils import count_layers

# lazily skip if torch isn't available in the environment
torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from probly.layers.torch import DropConnectLinear  # noqa: E402
from probly.transformation.dropconnect import dropconnect  # noqa: E402


class TestNetworkArchitectures:
    """Structure tests for different network architectures with DropConnect."""

    def test_linear_network_starts_with_linear(
        self, torch_model_small_2d_2d: nn.Sequential
    ) -> None:
        """
        If the network's first layer is Linear, DropConnect should *skip the first layer*
        and replace all subsequent Linear layers with DropConnectLinear.
        """
        p = 0.5
        model = dropconnect(torch_model_small_2d_2d, p)

        # original counts
        linear_orig = count_layers(torch_model_small_2d_2d, nn.Linear)
        seq_orig = count_layers(torch_model_small_2d_2d, nn.Sequential)

        # modified counts
        linear_mod = count_layers(model, nn.Linear)
        dc_mod = count_layers(model, DropConnectLinear)
        seq_mod = count_layers(model, nn.Sequential)

        # structure unchanged except Linear→DropConnectLinear replacements
        assert model is not None
        assert isinstance(model, type(torch_model_small_2d_2d))
        # first Linear remains, rest replaced
        assert dc_mod == max(0, linear_orig - 1)
        assert linear_mod == min(1, linear_orig)
        # Sequential container count unchanged
        assert seq_orig == seq_mod

    def test_conv_then_linear_network(
        self, torch_conv_linear_model: nn.Sequential
    ) -> None:
        """
        If the first layer is NOT Linear (e.g., Conv2d), *all* Linear layers
        should be replaced by DropConnectLinear.
        """
        p = 0.5
        model = dropconnect(torch_conv_linear_model, p)

        # original counts
        linear_orig = count_layers(torch_conv_linear_model, nn.Linear)
        conv_orig = count_layers(torch_conv_linear_model, nn.Conv2d)
        seq_orig = count_layers(torch_conv_linear_model, nn.Sequential)

        # modified counts
        linear_mod = count_layers(model, nn.Linear)
        dc_mod = count_layers(model, DropConnectLinear)
        conv_mod = count_layers(model, nn.Conv2d)
        seq_mod = count_layers(model, nn.Sequential)

        # all linears replaced; convs and containers unchanged
        assert isinstance(model, type(torch_conv_linear_model))
        assert linear_mod == 0
        assert dc_mod == linear_orig
        assert conv_mod == conv_orig
        assert seq_mod == seq_orig

    def test_custom_network_keeps_type(self, torch_custom_model: nn.Module) -> None:
        """Sanity: transformation preserves the top-level model type."""
        p = 0.5
        model = dropconnect(torch_custom_model, p)
        assert isinstance(model, type(torch_custom_model))


class TestPValues:
    """Verify that the probability p is propagated into DropConnect layers."""

    def test_p_value_in_linear_first_model(
        self, torch_model_small_2d_2d: nn.Sequential
    ) -> None:
        p = 0.3
        model = dropconnect(torch_model_small_2d_2d, p)

        for m in model.modules():
            if isinstance(m, DropConnectLinear):
                # DropConnectLinear exposes .p
                assert getattr(m, "p", None) == pytest.approx(p)

    def test_p_value_in_conv_model(self, torch_conv_linear_model: nn.Sequential) -> None:
        p = 0.2
        model = dropconnect(torch_conv_linear_model, p)

        for m in model.modules():
            if isinstance(m, DropConnectLinear):
                assert getattr(m, "p", None) == pytest.approx(p)
