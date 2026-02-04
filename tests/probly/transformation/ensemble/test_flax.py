"""Test ensemble flax nnx ensemble generation."""

from __future__ import annotations

import pytest

from probly.transformation import ensemble

pytest.importorskip("flax")
from flax import nnx
import jax
import jax.numpy as jnp

from tests.probly.flax_utils import count_layers


class TestEnsembleAttributes:
    """Test class for Ensemble attribute tests."""

    def test_ensemble_attributes_without_reset(self, flax_model_small_2d_2d) -> None:
        """Tests if the member attributes are inherited from the base model."""
        num_members = 2
        ensemble_model = ensemble(flax_model_small_2d_2d, num_members=2, reset_params=False)

        assert ensemble_model is not None
        assert isinstance(ensemble_model, nnx.List)
        assert len(ensemble_model) == num_members

        original_params = jax.tree_util.tree_leaves(flax_model_small_2d_2d)

        for member in ensemble_model:
            member_params = jax.tree_util.tree_leaves(member)

            assert original_params is not None
            assert member_params is not None
            assert jax.tree.structure(original_params) == jax.tree.structure(member_params)

            assert jax.tree_util.tree_reduce(
                lambda a, b: a & b,
                jax.tree_util.tree_map(lambda x, y: jnp.array_equal(x, y), original_params, member_params),
            )  # no difference

    @pytest.mark.skip(reason="not implemented yet")
    def test_ensemble_attributes_with_reset(self, flax_model_small_2d_2d) -> None:
        """Tests if the member attributes are not inherited from the base model."""
        num_members = 2
        ensemble_model = ensemble(flax_model_small_2d_2d, num_members=2, reset_params=True)

        assert ensemble_model is not None
        assert isinstance(ensemble_model, nnx.List)
        assert len(ensemble_model) == num_members

        original_params = jax.tree_util.tree_leaves(flax_model_small_2d_2d)

        for member in ensemble_model:
            member_params = jax.tree_util.tree_leaves(member)

            assert original_params is not None
            assert member_params is not None
            assert jax.tree.structure(original_params) == jax.tree.structure(member_params)

            assert jax.tree_util.tree_reduce(
                lambda a, b: a | b,
                jax.tree_util.tree_map(lambda x, y: ~jnp.allclose(x, y), original_params, member_params),
            )  # any difference


class TestEnsembleGeneration:
    """Test class for Ensemble generation tests."""

    @pytest.mark.parametrize(
        "model_fixture",
        [
            "flax_model_small_2d_2d",
            "flax_conv1d_linear_model",
            "flax_conv_linear_model",
            "flax_conv3d_linear_model",
            "flax_regression_model_1d",
            "flax_regression_model_2d",
        ],
    )
    def test_fixtures(
        self,
        request: pytest.FixtureRequest,
        model_fixture: str,
    ) -> None:
        model = request.getfixturevalue(model_fixture)
        num_members = 5

        ensemble_model = ensemble(model, num_members=num_members, reset_params=False)

        count_linear_original = count_layers(model, nnx.Linear)
        count_conv_original = count_layers(model, nnx.Conv)
        count_sequential_original = count_layers(model, nnx.Sequential)
        count_module_original = count_layers(model, nnx.Module)

        assert ensemble_model is not None
        assert isinstance(ensemble_model, nnx.List)
        assert len(ensemble_model) == num_members

        for member in ensemble_model:
            count_linear_member = count_layers(member, nnx.Linear)
            count_conv_member = count_layers(member, nnx.Conv)
            count_sequential_member = count_layers(member, nnx.Sequential)
            count_module_member = count_layers(member, nnx.Module)
            assert count_linear_member == count_linear_original
            assert count_conv_member == count_conv_original
            assert count_sequential_member == count_sequential_original
            assert count_module_member == count_module_original

    def test_custom_model(self, flax_custom_model) -> None:
        num_members = 2
        ensemble_model = ensemble(flax_custom_model, num_members=num_members, reset_params=False)

        count_module_original = count_layers(flax_custom_model, nnx.Module)

        assert ensemble_model is not None
        assert isinstance(ensemble_model, nnx.List)
        assert len(ensemble_model) == num_members

        for member in ensemble_model:
            count_module_member = count_layers(member, nnx.Module)
            assert count_module_member == count_module_original

    def test_not_implemented_error_with_reset(self, flax_model_small_2d_2d) -> None:
        num_members = 2

        msg = "resetting parameters of flax models is not supported yet."
        with pytest.raises(NotImplementedError, match=msg):
            ensemble(flax_model_small_2d_2d, num_members=num_members, reset_params=True)


class TestEnsembleCalls:
    """Test class for ensemble model calls."""

    def test_ensemble_flax_custom_model_call(self, flax_custom_model) -> None:
        num_members = 2
        ensemble_model = ensemble(flax_custom_model, num_members=num_members, reset_params=False)

        x = jnp.ones((2, 1, 10))
        custom_model_out = flax_custom_model(x)

        for member in ensemble_model:
            member_out = member(x)
            assert custom_model_out.shape == member_out.shape
            assert jnp.equal(custom_model_out, member_out).all()  # no parameter reset

    @pytest.mark.skip(reason="not implemented yet")
    def test_ensemble_flax_custom_model_call_with_reset(self, flax_custom_model) -> None:
        num_members = 2
        ensemble_model = ensemble(flax_custom_model, num_members=num_members, reset_params=True)

        x = jnp.ones((2, 1, 10))
        custom_model_out = flax_custom_model(x)

        for member in ensemble_model:
            member_out = member(x)
            assert custom_model_out.shape == member_out.shape
            assert not jnp.equal(custom_model_out, member_out).all()  # parameter reset
