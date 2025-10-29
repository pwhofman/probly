"""Tests for torch ensemble generation."""
from __future__ import annotations 

import pytest
from probly.transformation import ensemble 

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

class TestGenerateTorchEnsemble:
    """Test class for torch ensemble generation."""

    def test_generate_torch_ensemble_creates_n_models(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        original_model = torch_model_small_2d_2d
        n = 3

        new_models = ensemble(original_model, n_members=n)

        #checks if it is a modulelist
        assert isinstance(new_models, nn.ModuleList)

        #checks if there are exactly n modules
        assert len(new_models) == n
        
     
    def test_generate_torch_ensemble_creates_zero_models(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """n_members=0 should return an empty ModuleList."""
        original_model = torch_model_small_2d_2d
        new_models = ensemble(original_model, n_members=0)

        assert isinstance(new_models, nn.ModuleList)
        assert len(new_models) == 0 #should be empty
          
    def test_different_obj(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Ensure ensemble members are different objects."""  
        original_model = torch_model_small_2d_2d

        new_models = ensemble(original_model, n_members=2)
        a, b = new_models

        #different objects
        assert a is not original_model
        assert b is not original_model
        assert a is not b

    def test_not_shared_params(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """parameter tensors must not share storage."""
        original_model = torch_model_small_2d_2d

        new_models = ensemble(original_model, n_members=2)
        a, b = new_models

        #between members
        for pa, pb in zip(a.parameters(), b.parameters()):
            assert pa.detach().data_ptr() != pb.detach().data_ptr()

        #between member and original_model
        for po, pa in zip(original_model.parameters(), a.parameters()):
            assert po.detach().data_ptr() != pa.detach().data_ptr()

    def test_mutating_one_member_does_not_affect_the_other(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        """Changing parameters in one member must not change parameters in another member."""
        original_model = torch_model_small_2d_2d
        new_models = ensemble(original_model, n_members=2)
        a, b = new_models
        b_first_before = next(b.parameters()).detach().clone()

        with torch.no_grad():
            next(a.parameters()).add_(1.2345)

        assert torch.allclose(b_first_before, next(b.parameters()).detach())

 
  




