"""Tests for torch DropConnect models."""

from __future__ import annotations

import pytest

from probly.transformation import dropconnect
from tests.probly.torch_utils import count_layers

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from probly.layers.torch import DropConnectLinear  # noqa: E402


class TestNetworkArchitectures:
    def test_linear_network_replaces_linear_layers(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        p = 0.5
        model = dropconnect(torch_model_small_2d_2d, p)

        # count original layers
        count_linear_original = count_layers(torch_model_small_2d_2d, nn.Linear)
        count_sequential_original = count_layers(torch_model_small_2d_2d, nn.Sequential)

        # count modified layers
        count_dropconnect_modified = count_layers(model, DropConnectLinear)
        count_linear_modified = count_layers(model, nn.Linear)
        count_sequential_modified = count_layers(model, nn.Sequential)

        # checks layer counts and modified skips first layer
        assert model is not None
        assert isinstance(model, type(torch_model_small_2d_2d))
        assert count_dropconnect_modified == (count_linear_original - 1)
        assert count_linear_modified == 1
        assert count_sequential_original == count_sequential_modified

    def test_p_value_propagation(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        p = 0.3
        model = dropconnect(torch_model_small_2d_2d, p)

        for m in model.modules():
            if isinstance(m, DropConnectLinear):
                assert pytest.approx(m.p) == p
