"""Tests for torch ensemble generation."""

from __future__ import annotations

import pytest

from probly.method.ensemble import ensemble

pytest.importorskip("torch")
import torch
from torch import nn

from probly.layers.torch import BatchEnsembleLinear, DropConnectLinear
from probly.method.batchensemble import batchensemble
from probly.method.dropconnect import dropconnect


def assert_pairwise_different(tensors: list[torch.Tensor]) -> None:
    """Assert that no two of ``tensors`` are equal."""
    for i in range(len(tensors)):
        for j in range(i + 1, len(tensors)):
            assert not torch.equal(tensors[i], tensors[j]), f"tensors {i} and {j} are identical"


def assert_members_are_independent(base: nn.Module, members: nn.ModuleList) -> None:
    """Every randomly initialized parameter differs between the members and from the base model.

    Parameters that a module initializes to a constant (LayerNorm gains, zero attention biases) hold
    that constant in every member.
    """
    member_params = [dict(member.named_parameters()) for member in members]
    for name, base_param in base.named_parameters():
        values = [params[name].detach() for params in member_params]
        if base_param.numel() > 1 and base_param.detach().std() > 0:
            assert_pairwise_different([base_param.detach(), *values])
        else:
            assert all(torch.equal(value, base_param.detach()) for value in values), name


class TransformerClassifier(nn.Module):
    """A transformer encoder with a classification head and a learned class token."""

    def __init__(self) -> None:
        """Build the encoder, the token and the head."""
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, 8))
        self.enc = nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16, batch_first=True)
        self.head = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        return self.head(self.enc(tokens)[:, 0])


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


class TestResetMembersAreIndependent:
    """With reset_params=True, every trainable parameter is freshly initialized per member."""

    def test_transformer_layer_attention_weights_differ(self) -> None:
        """Every member used to share the base model's attention in-projection."""
        torch.manual_seed(0)
        base = nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16, batch_first=True)

        members = ensemble(base, num_members=3, reset_params=True)

        assert_pairwise_different(
            [base.self_attn.in_proj_weight.detach(), *(m.self_attn.in_proj_weight.detach() for m in members)]
        )
        assert_members_are_independent(base, members)

    def test_model_with_attention_and_class_token(self) -> None:
        torch.manual_seed(0)
        base = TransformerClassifier()

        members = ensemble(base, num_members=3, reset_params=True)

        assert_members_are_independent(base, members)
        assert_pairwise_different([base.cls_token.detach(), *(m.cls_token.detach() for m in members)])

    def test_members_make_different_predictions(self) -> None:
        torch.manual_seed(0)
        members = ensemble(TransformerClassifier(), num_members=3, reset_params=True).eval()
        x = torch.randn(4, 5, 8)

        with torch.no_grad():
            outputs = [member(x) for member in members]

        assert_pairwise_different(outputs)
        assert all(torch.isfinite(out).all() for out in outputs)

    def test_dropconnect_members_differ(self) -> None:
        """DropConnectLinear has its own reset scheme; a reset must still make members differ."""
        torch.manual_seed(0)
        base = dropconnect(nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 3)))

        members = ensemble(base, num_members=3, reset_params=True)

        assert any(isinstance(layer, DropConnectLinear) for layer in members[0])
        assert_members_are_independent(base, members)

    def test_batchensemble_members_differ(self) -> None:
        torch.manual_seed(0)
        base = batchensemble(nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)), num_members=2)

        members = ensemble(base, num_members=3, reset_params=True)

        assert any(isinstance(layer, BatchEnsembleLinear) for layer in members[0])
        assert_members_are_independent(base, members)

    def test_batchensemble_members_differ_with_use_base_weights(self) -> None:
        """The shared weight built with use_base_weights=True must still be redrawn per member."""
        torch.manual_seed(0)
        base = batchensemble(
            nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)),
            num_members=2,
            use_base_weights=True,
        )

        members = ensemble(base, num_members=3, reset_params=True)

        assert any(isinstance(layer, BatchEnsembleLinear) for layer in members[0])
        assert_members_are_independent(base, members)

    def test_single_member_differs_from_base(self) -> None:
        torch.manual_seed(0)
        base = TransformerClassifier()

        (member,) = ensemble(base, num_members=1, reset_params=True)

        assert not torch.equal(member.enc.self_attn.in_proj_weight, base.enc.self_attn.in_proj_weight)
        assert not torch.equal(member.cls_token, base.cls_token)

    def test_without_reset_members_copy_every_parameter(self) -> None:
        torch.manual_seed(0)
        base = TransformerClassifier()

        members = ensemble(base, num_members=2, reset_params=False)

        for member in members:
            for (name, param), base_param in zip(member.named_parameters(), base.parameters(), strict=True):
                assert torch.equal(param, base_param), name
