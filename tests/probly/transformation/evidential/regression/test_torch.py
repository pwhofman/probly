from __future__ import annotations

import pytest

from probly.transformation.evidential.regression import evidential_regression
from tests.probly.torch_utils import count_layers
from probly.layers.torch import NormalInverseGammaLinear

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402




class TestNetworkArchitectures:
    """Test class for different network architectures."""

    @pytest.mark.parametrize('torch_regression_model_name',['torch_regression_model_1d', 'torch_regression_model_2d'])
    def test_linear_network_with_last_linear(self, torch_regression_model_name, request) -> None:
        """Tests if a model incorporates a normal invers gamma layer correctly in exchange for the last linear layer.

        This function verifies that:
        - A normal invers gamma layer substitutes the last linear layer in the model.
        - The structure of the model remains unchanged except for the normal invers gamma layer.

        It performs counts and asserts to ensure the modified model adheres to expectations.

        Parameters:
            torch_regression_model_1d: The torch model to be tested, specified as a sequential model.

        Raises:
            AssertionError: If the structure of the model differs in an unexpected manner or if the normal invers gamma layer is not
            inserted correctly after linear layers.
        """
        torch_regression_model = request.getfixturevalue(torch_regression_model_name)
        model = evidential_regression(torch_regression_model)

        # count number of nn.Linear layers in original model
        count_linear_original = count_layers(torch_regression_model, nn.Linear)
        # count number of nn.NormalInverseGammaLinear layers in original model
        count_nig_original = count_layers(torch_regression_model, NormalInverseGammaLinear)
        # count number of nn.Sequential layers in original model
        count_sequential_original = count_layers(torch_regression_model, nn.Sequential)
        # count number of nn.Module in original model
        count_all_modules_original = count_layers(torch_regression_model, nn.Module)

        # count number of nn.Linear layers in modified model
        count_linear_modified = count_layers(model, nn.Linear)
        # count number of nn.NormalInverseGammaLinear layers in modified model
        count_nig_modified = count_layers(model, NormalInverseGammaLinear)
        # count number of nn.Sequential layers in modified model
        count_sequential_modified = count_layers(model, nn.Sequential)
        # count number of nn.Module in modified model
        count_all_modules_modified = count_layers(model, nn.Module)

        # check that the model is not modified except for the nig layer
        assert model is not None
        assert isinstance(model, type(torch_regression_model))
        assert (count_linear_original - 1) == count_linear_modified
        assert count_nig_modified == 1
        assert count_nig_original == 0
        assert count_sequential_original == count_sequential_modified
        assert count_all_modules_original == count_all_modules_modified

        def count_linear_layers_until_nig(model: nn.Module) -> int:
            s=0
            for m in model.children():
                if isinstance(m, nn.Linear):
                    s+=1
                if isinstance(m, NormalInverseGammaLinear):
                    return s
            return s
        
        #last layer to be nig
        assert (count_linear_layers_until_nig(torch_regression_model)-1) == count_linear_layers_until_nig(model)
