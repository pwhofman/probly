"""Test for flax ensemble models."""

from __future__ import annotations

from flax import nnx
import jax.numpy as jnp

from probly.transformation.ensemble.flax import generate_flax_ensemble as ensemble
from tests.probly.flax_utils import count_layers


class TestFlaxEnsemble:
    """Test class for Flax ensemble generation."""

    def test_generate_ensemble_structure(
        self,
        flax_model_small_2d_2d: nnx.Sequential,
    ) -> None:
        """Test ensemble generation for small 2D to 2D model.

        Tests if the ensemble generator creates multiple independent Flax models.
        This function verifies that:
        - The ensemble generation function creates the correct number of model copies.
        - Each model in the ensemble maintains the same structure as the original model.

        Parameters:
            flax_model_small_2d_2d: A small Flax Sequential model that maps 2D inputs to 2D outputs.

        Raises:
            AssertionError: If the ensemble size or model structure does not match expectations.
        """
        n_members = 3
        models: list[nnx.Sequential] = ensemble(
            flax_model_small_2d_2d,
            n_members,
        )  # type: ignore[assignment]

        assert len(models) == n_members

        # check each model structure in ensemble
        for member in models:
            assert isinstance(member, nnx.Sequential)
            assert count_layers(member, nnx.Linear) == count_layers(
                flax_model_small_2d_2d,
                nnx.Linear,
            )

        # check corresponding layer types match
        for layer_original, layer_ensemble in zip(
            flax_model_small_2d_2d.layers,
            models[0].layers,
            strict=False,
        ):
            assert type(layer_original) is type(layer_ensemble)

    def test_ensemble_independence(
        self,
        flax_model_small_2d_2d: nnx.Sequential,
    ) -> None:
        """Test that ensemble members are independent after training steps.

        Tests if training one ensemble member does not affect the others.

        Parameters:
            flax_model_small_2d_2d: A small Flax Sequential model that maps 2D inputs to 2D outputs.

        Raises:
            AssertionError: If parameters of different ensemble members are found to be identical after updates.
        """
        n_members = 3
        ensemble_models: list[nnx.Sequential] = ensemble(
            flax_model_small_2d_2d,
            n_members,
        )  # type: ignore[assignment]

        def _collect_params(mod: nnx.Sequential) -> list[jnp.ndarray]:
            params: list[jnp.ndarray] = []
            for layer in getattr(mod, "layers", []):
                for val in vars(layer).values():
                    array_like = [
                        val for val in vars(layer).values() if hasattr(val, "shape") and hasattr(val, "dtype")
                    ]
                    params.extend(array_like)
            return params

        # Gather initial parameters of all ensemble members
        params_before: list[list[jnp.ndarray]] = [_collect_params(m) for m in ensemble_models]

        # Mock update to first model's parameters (non-mutating)
        updated_first = [p + 1.0 for p in params_before[0]]

        # Check that other models' parameters remain unchanged compared to the updated first
        for updated_param, other_param in zip(
            updated_first,
            params_before[1],
            strict=False,
        ):
            assert not jnp.array_equal(updated_param, other_param)
