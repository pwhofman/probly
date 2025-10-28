"""Test for flax ensemble models."""

from __future__ import annotations

import pytest

from probly.transformation import ensemble
from tests.probly.flax_utils import count_layers

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402


class TestEnsembleArchitectures:
    """Test class for different ensemble network architectures."""

    def test_linear_network_ensemble(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if a linear model is correctly converted into an ensemble.

        Verifies that:
        - The ensemble transformation produces multiple model instances.
        - The overall structure of each submodel is identical to the original.
        - The ensemble wrapper preserves sequential model structure.

        Parameters:
            flax_model_small_2d_2d: The flax model to be tested.

        Raises:
            AssertionError: If the ensemble structure differs unexpectedly.
        """
        n = 3
        model = ensemble(flax_model_small_2d_2d, n=n)

        # Check the overall model type
        assert model is not None
        assert hasattr(model, "members"), "Ensemble model should have `members` attribute."
        assert len(model.members) == n, f"Expected {n} ensemble members, got {len(model.members)}."

        # Compare layer counts between original and ensemble members
        count_linear_original = count_layers(flax_model_small_2d_2d, nnx.Linear)
        for member in model.members:
            assert isinstance(member, type(flax_model_small_2d_2d))
            count_linear_member = count_layers(member, nnx.Linear)
            assert count_linear_member == count_linear_original

    def test_convolutional_network_ensemble(self, flax_conv_linear_model: nnx.Sequential) -> None:
        """Tests the ensemble transformation on a convolutional + linear model."""
        n = 5
        model = ensemble(flax_conv_linear_model, n=n)

        assert model is not None
        assert hasattr(model, "members")
        assert len(model.members) == n

        # Check each member for identical structure
        count_conv_original = count_layers(flax_conv_linear_model, nnx.Conv)
        count_linear_original = count_layers(flax_conv_linear_model, nnx.Linear)
        for member in model.members:
            assert isinstance(member, type(flax_conv_linear_model))
            assert count_layers(member, nnx.Conv) == count_conv_original
            assert count_layers(member, nnx.Linear) == count_linear_original

    def test_custom_model_ensemble(self, flax_custom_model: nnx.Module) -> None:
        """Tests custom model ensemble generation."""
        n = 4
        model = ensemble(flax_custom_model, n=n)

        assert isinstance(model, type(ensemble(flax_custom_model, n=1)))  # same wrapper type
        assert hasattr(model, "members")
        assert all(isinstance(m, type(flax_custom_model)) for m in model.members)


class TestEnsembleConsistency:
    """Test class for internal ensemble consistency."""

    def test_member_independence(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Ensures ensemble members are independent (not referencing same params)."""
        n = 3
        model = ensemble(flax_model_small_2d_2d, n=n)

        # Get references to params of first layer in each member
        first_params = [id(next(iter(m.parameters().values()))) for m in model.members]
        # Assert that no two members share same parameter references
        assert len(set(first_params)) == n, "Ensemble members share parameter references!"

    def test_forward_output_shape(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Ensures ensemble forward outputs are aggregated correctly."""
        import jax
        import jax.numpy as jnp

        n = 2
        model = ensemble(flax_model_small_2d_2d, n=n)
        x = jnp.ones((1, 2))  # dummy input

        y = model(x)
        # Assuming ensemble returns an averaged or stacked output
        assert y is not None
        assert y.shape[0] in (1, n), "Unexpected ensemble output shape."


class TestNumMembers:
    """Test class for verifying number of ensemble members."""

    def test_different_ensemble_sizes(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Checks that different n values correctly control ensemble size."""
        for n in [1, 2, 5]:
            model = ensemble(flax_model_small_2d_2d, n=n)
            assert len(model.members) == n

    def test_invalid_n_raises(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests that invalid ensemble sizes raise proper errors."""
        with pytest.raises(ValueError):
            _ = ensemble(flax_model_small_2d_2d, n=0)
