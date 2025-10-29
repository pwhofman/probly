"""Test for Flax ensemble generation."""

from __future__ import annotations
import pytest
import jax
import jax.numpy as jnp
from flax import nnx

from probly.transformation.flax_ensemble import generate_flax_ensemble


class DummyModel(nnx.Module):
    def __init__(self, features: int):
        self.dense = nnx.Linear(features=features)

    def __call__(self, x):
        return self.dense(x)


class TestGenerateFlaxEnsemble:
    """Tests for the low-level Flax ensemble generator."""

    def test_generate_ensemble_shapes_and_count(self) -> None:
        """Checks that correct number of ensemble members are created."""
        model = DummyModel(features=4)
        input_shape = (1, 3)
        n = 3
        ensemble = generate_flax_ensemble(model, n_members=n, input_shape=input_shape)

        assert isinstance(ensemble, list)
        assert len(ensemble) == n

        for submodel, params in ensemble:
            assert isinstance(submodel, DummyModel)
            assert isinstance(params, dict)
            # verify params contain dense layer weights
            assert "dense" in params

    def test_different_initializations(self) -> None:
        """Ensures each ensemble member is initialized independently."""
        model = DummyModel(features=2)
        input_shape = (1, 2)
        n = 3
        ensemble = generate_flax_ensemble(model, n_members=n, input_shape=input_shape)

        weights = [list(params["dense"].values())[0] for _, params in ensemble]
        # check at least one pair of weights differs
        diffs = [not jnp.allclose(weights[i], weights[j]) for i in range(n) for j in range(i + 1, n)]
        assert any(diffs), "All ensemble members were initialized identically!"

    def test_zero_members_raises(self) -> None:
        """Ensure invalid ensemble sizes raise an error."""
        model = DummyModel(features=2)
        with pytest.raises(ValueError):
            _ = generate_flax_ensemble(model, n_members=0, input_shape=(1, 2))