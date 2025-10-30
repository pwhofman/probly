'''Test for torch ensemble module'''
#Importe von ensemble torch und test_torch by dropout
from __future__ import annotations
import pytest
from probly.transformation import ensemble
from tests.probly.torch_utils import count_layers
torch = pytest.importorskip("torch")
from pytraverse import CLONE, lazydispatch_traverser, traverse
from torch import nn  # noqa: E402 
reset_traverser = lazydispatch_traverser[object](name="reset_traverser")

class TestEnsembleModule:
    def test_return_type_and_length(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        k = 3
        ens = ensemble(torch_model_small_2d_2d, n_members=k)
        assert isinstance(ens, nn.ModuleList)
        assert len(ens) == k

    def test_number_of_members(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        n_members = 3
        model = ensemble(torch_model_small_2d_2d, n_members=n_members)
        assert len(model) == n_members

    def test_returns_module_list(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        n_members = 3
        model = ensemble(torch_model_small_2d_2d, n_members=n_members)
        assert isinstance(model, nn.ModuleList)

    def test_layer_counts_scale_linear(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        #Checks if number of layers are correct for linear 
        k = 3
        base = torch_model_small_2d_2d
        ens = ensemble(base, n_members=k)

        assert count_layers(ens, nn.Linear) == k * count_layers(base, nn.Linear)
        assert count_layers(ens, nn.Sequential) == k * count_layers(base, nn.Sequential)
        assert count_layers(ens, nn.Dropout) == k * count_layers(base, nn.Dropout)

    def test_layer_counts_scale_conv(self, torch_conv_linear_model: nn.Sequential) -> None:
        #checks if number of layers is correct for convolution
        k = 4
        base = torch_conv_linear_model
        ens = ensemble(base, n_members=k)
        assert count_layers(ens, nn.Linear) == k * count_layers(base, nn.Linear)
        assert count_layers(ens, nn.Conv2d) == k * count_layers(base, nn.Conv2d)
        assert count_layers(ens, nn.Sequential) == k * count_layers(base, nn.Sequential)

    def test_deep_copy(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        k = 2
        ens = ensemble(torch_model_small_2d_2d, n_members=k)
        p0 = next(ens[0].parameters())
        p1 = next(ens[1].parameters())
        #data_ptr returns address 
        assert p0.data_ptr() != p1.data_ptr()
   