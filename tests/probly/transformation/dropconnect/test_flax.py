
from __future__ import annotations

import pytest

from probly.transformation import dropout
from tests.probly.flax_utils import count_layers
flax = pytest.importorskip("flax")
from flax import nnx
from probly.layers.flax import DropConnectDense 
from probly.transformation.dropconnect import dropconnect

class TestNetworkArchitectures:
    """Structure tests for different network architectures with DropConnect."""

    def test_linear_network_starts_with_linear(
        self, flax_model_small_2d_2d: nnx.Sequential
    ) -> None:
        """
        If the network's first layer is Linear, DropConnect should *skip the first layer*
        and replace all subsequent Linear layers with DropConnectLinear.
        """
        p = 0.5
        model = dropconnect(flax_model_small_2d_2d, p)

        # original counts
        linear_orig = count_layers(flax_model_small_2d_2d, nnx.Linear)
        seq_orig = count_layers(flax_model_small_2d_2d, nnx.Sequential)

        # modified counts
        linear_mod = count_layers(model, nnx.Linear)
        dc_mod = count_layers(model, DropConnectDense)
        seq_mod = count_layers(model, nnx.Sequential)

        # structure unchanged except Linear→DropConnectLinear replacements
        assert model is not None
        assert isinstance(model, type(flax_model_small_2d_2d))
        # first Linear remains, rest replaced
        assert dc_mod == max(0, linear_orig - 1)
        assert linear_mod == min(1, linear_orig)
        # Sequential container count unchanged
        assert seq_orig == seq_mod

    def test_conv_then_linear_network(
        self, torch_conv_linear_model: nnx.Sequential
    ) -> None:
        """
        If the first layer is NOT Linear (e.g., Conv2d), *all* Linear layers
        should be replaced by DropConnectLinear.
        """
        p = 0.5
        model = dropconnect(torch_conv_linear_model, p)

        # original counts
        linear_orig = count_layers(torch_conv_linear_model, nnx.Linear)
        conv_orig = count_layers(torch_conv_linear_model, nnx.Conv)
        seq_orig = count_layers(torch_conv_linear_model, nnx.Sequential)

        # modified counts
        linear_mod = count_layers(model, nnx.Linear)
        dc_mod = count_layers(model, DropConnectDense)
        conv_mod = count_layers(model, nnx.Conv)
        seq_mod = count_layers(model, nnx.Sequential)

        # all linears replaced; convs and containers unchanged
        assert isinstance(model, type(torch_conv_linear_model))
        assert linear_mod == 0
        assert dc_mod == linear_orig
        assert conv_mod == conv_orig
        assert seq_mod == seq_orig

