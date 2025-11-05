"""Tests for flax ensemble transformation."""

from __future__ import annotations

import pytest

pytest.importorskip("flax")

from flax import nnx
import numpy as np

from probly.transformation.ensemble import ensemble


class TestEnsembleArchitecture:
    """Test class for ensemble architecture tests."""

    def test_ensemble_count(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if the ensemble transformation returns the correct number of models.

        Parameters:
            flax_model_small_2d_2d: The flax model to be tested.
        """
        n_members = 5

        model_list = ensemble(flax_model_small_2d_2d, n_members=n_members)

        # Check if the returned object is a list
        assert isinstance(model_list, list)

        # Check if the list has the correct length
        assert len(model_list) == n_members

    def test_models_are_clones(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if the ensemble transformation returns clones of the original model.

        Parameters:
            flax_model_small_2d_2d: The flax model to be tested.
        """
        n_members = 2

        model_list = ensemble(flax_model_small_2d_2d, n_members=n_members)

        # Check that the new models are not the original object
        assert model_list[0] is not flax_model_small_2d_2d
        assert model_list[1] is not flax_model_small_2d_2d

        # Check that the new models are different from each other
        assert model_list[0] is not model_list[1]


class TestEnsembleParameters:
    """Test class for ensemble parameters tests."""

    def test_models_are_reset(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if the parameters of the new models are different from the original and from each other.

        Parameters:
            flax_model_small_2d_2d: The flax model to be tested.
        """
        n_members = 3
        original_weights = flax_model_small_2d_2d.layers[0].kernel.value

        model_list = ensemble(flax_model_small_2d_2d, n_members=n_members)

        # Get the weights from the new models
        weights_model_1 = model_list[0].layers[0].kernel.value
        weights_model_2 = model_list[1].layers[0].kernel.value

        # Check if the new weights are DIFFERENT from the original weights
        assert not np.array_equal(weights_model_1, original_weights)
        assert not np.array_equal(weights_model_2, original_weights)

        # Check if the new weights are DIFFERENT from each other
        assert not np.array_equal(weights_model_1, weights_model_2)
