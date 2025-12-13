"""Tests for torch ensemble generation."""

from __future__ import annotations

import pytest

from probly.transformation import ensemble

pytest.importorskip("torch")
import torch
from torch import nn


class TestGenerateTorchEnsemble:
    """Test class for torch ensemble generation."""

    def test_generate_torch_ensemble_creates_n_models(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        original_model = torch_model_small_2d_2d
        n = 3

        new_models = ensemble(original_model, num_members=n)

        # checks if it is a modulelist
        assert isinstance(new_models, nn.ModuleList)

        # checks if there are exactly n modules
        assert len(new_models) == n

    def test_generate_torch_ensemble_creates_zero_models(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """n_members=0 should return an empty ModuleList."""
        original_model = torch_model_small_2d_2d
        new_models = ensemble(original_model, num_members=0)

        assert isinstance(new_models, nn.ModuleList)
        assert len(new_models) == 0  # should be empty

    def test_different_obj(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Ensure ensemble members are different objects."""
        original_model = torch_model_small_2d_2d

        new_models = ensemble(original_model, num_members=2)
        a, b = new_models

        # different objects
        assert a is not original_model
        assert b is not original_model
        assert a is not b

    def test_not_shared_params(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Parameter tensors must not share storage."""
        original_model = torch_model_small_2d_2d

        new_models = ensemble(original_model, num_members=2)
        a, b = new_models

        # between members
        for pa, pb in zip(a.parameters(), b.parameters(), strict=False):
            assert pa.detach().data_ptr() != pb.detach().data_ptr()

        # between member and original_model
        for po, pa in zip(original_model.parameters(), a.parameters(), strict=False):
            assert po.detach().data_ptr() != pa.detach().data_ptr()

    def test_mutating_one_member_does_not_affect_the_other(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Changing parameters in one member must not change parameters in another member."""
        original_model = torch_model_small_2d_2d
        new_models = ensemble(original_model, num_members=2)
        a, b = new_models
        b_first_before = next(b.parameters()).detach().clone()

        with torch.no_grad():
            next(a.parameters()).add_(1.2345)

        assert torch.allclose(b_first_before, next(b.parameters()).detach())

    def test_two_ensembles_are_different(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Tests if two ensembles created from the same base are different (due to RNG)."""
        original_model = torch_model_small_2d_2d

        # Create ensemble 1
        ensemble_1 = ensemble(original_model, num_members=1)
        # Get parameter of the only member
        param_e1 = next(ensemble_1[0].parameters()).detach().clone()

        # Create ensemble 2
        ensemble_2 = ensemble(original_model, num_members=1)
        param_e2 = next(ensemble_2[0].parameters()).detach().clone()

        # parameters should be different because of random initialization
        assert not torch.equal(param_e1, param_e2), "Ensembles should differ due to RNG, but parameters are identical."

    def test_forward_passes_shape(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Ensure that the ensemble forward pass produces outputs of expected shape."""
        original_model = torch_model_small_2d_2d
        n_members = 4
        batch_size = 5
        input_dim = 2

        new_models = ensemble(original_model, num_members=n_members)

        # Create a dummy input
        dummy_input = torch.randn(batch_size, input_dim)

        # Collect outputs from each ensemble member
        outputs = [model(dummy_input) for model in new_models]

        # Check that each output has the correct shape
        for output in outputs:
            assert output.shape == (batch_size, 2)  # Assuming original model output dim is 2

    def test_output_types(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Ensure that the outputs of ensemble members are tensors."""
        original_model = torch_model_small_2d_2d
        n_members = 3
        batch_size = 4
        input_dim = 2

        new_models = ensemble(original_model, num_members=n_members)

        # Create a dummy input
        dummy_input = torch.randn(batch_size, input_dim)

        # Collect outputs from each ensemble member
        outputs = [model(dummy_input) for model in new_models]

        # Check that each output is a tensor
        for output in outputs:
            assert isinstance(output, torch.Tensor)

    def test_no_params_uses_original_params(self) -> None:
        """If the model has no parameters, ensure the ensemble members are still created correctly."""

        class NoParamModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2

        original_model = NoParamModel()
        n_members = 3
        batch_size = 4
        input_dim = 2

        new_models = ensemble(original_model, num_members=n_members)

        # Create a dummy input
        dummy_input = torch.randn(batch_size, input_dim)

        # Collect outputs from each ensemble member
        outputs = [model(dummy_input) for model in new_models]

        # Check that each output is correct
        for output in outputs:
            assert torch.allclose(output, dummy_input * 2)

    def test_no_reset_params_preserves_values(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Ensure that with reset_params=False, parameters and outputs are preserved."""
        original_model = torch_model_small_2d_2d

        # save original parameter values
        original_param_value = next(original_model.parameters()).detach().clone()

        # create ensemble with reset_params=False
        new_models = ensemble(original_model, num_members=2, reset_params=False)
        a, b = new_models

        # parameter values must be the same as original
        param_a = next(a.parameters()).detach()
        assert torch.allclose(
            original_param_value,
            param_a,
        ), "Parameter should have the same values as the original with reset_params=False."

        # outputs must be identical
        dummy_input = torch.randn(4, 2)
        output_a = a(dummy_input)
        output_b = b(dummy_input)

        assert torch.allclose(output_a, output_b), "Outputs must be identical when parameters are not reset."
