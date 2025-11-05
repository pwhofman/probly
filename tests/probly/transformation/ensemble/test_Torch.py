"""Tests for torch ensemble models."""

from __future__ import annotations
import pytest

torch = pytest.importorskip("torch")
from torch import nn, Tensor

from probly.transformation.ensemble import ensemble

class TestTorchEnsemble: 

    def test_invalid_members(self, torch_model_small_2d_2d: nn.Module) -> None: 
        
        with pytest.raises(ValueError, match="n_members must be >= 1"):
            ensemble(torch_model_small_2d_2d, n_members=0)

    def test_returning_modulelist(self, torch_model_small_2d_2d: nn.Module) -> None: 

        model = ensemble(torch_model_small_2d_2d, n_members=3)
        assert isinstance(model, nn.ModuleList)

    def test_correct_number_of_members(self, torch_model_small_2d_2d: nn.Module) -> None: 

        n_members = 5
        model_list = ensemble(torch_model_small_2d_2d, n_members=n_members)

        assert len(model_list) == n_members, f"Expected {n_members} members, got {len(model_list)}"
        
    def test_each_member_is_model(self, torch_model_small_2d_2d: nn.Module) -> None:

        n_members = 3
        model_list = ensemble(torch_model_small_2d_2d, n_members=n_members)
        for m in model_list:
            assert isinstance(m, nn.Module)

    def test_members_have_independent_parameters(self, torch_model_small_2d_2d: nn.Module) -> None:
        
        model_list = ensemble(torch_model_small_2d_2d, n_members=2)
        par1 = list(model_list[0].parameters())[0].detach().clone()
        par2 = list(model_list[1].parameters())[0].detach().clone()

        assert not torch.equal(par1, par2), "Members should have independent parameters"

