"""Tests for flax ensemble transformation."""

from __future__ import annotations

import pytest

pytest.importorskip("flax")

from flax import nnx
import jax.numpy as jnp
import numpy as np

from probly.transformation.ensemble import ensemble


class TestEnsembleArchitecture:
    """Test class for ensemble architecture tests."""

    def test_ensemble_count(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if the ensemble transformation returns the correct number of models.

        Parameters:
            flax_model_small_2d_2d: The flax model to be tested.
        """
        num_members = 5

        model_list = ensemble(flax_model_small_2d_2d, num_members=num_members)

        # Check if the returned object is a list
        assert isinstance(model_list, list)

        # Check if the list has the correct length
        assert len(model_list) == num_members

    def test_ensemble_count_zero(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if num_members=0 returns an empty list (äquivalent zu Torch)."""
        model_list = ensemble(flax_model_small_2d_2d, num_members=0)

        assert isinstance(model_list, list)
        assert len(model_list) == 0  # should be empty

    def test_models_are_clones(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if the ensemble transformation returns clones of the original model.

        Parameters:
            flax_model_small_2d_2d: The flax model to be tested.
        """
        num_members = 2

        model_list = ensemble(flax_model_small_2d_2d, num_members=num_members)

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
        num_members = 3

        original_weights = flax_model_small_2d_2d.layers[0].kernel.value

        model_list = ensemble(flax_model_small_2d_2d, num_members=num_members)

        # Get the weights from the new models
        weights_model_1 = model_list[0].layers[0].kernel.value
        weights_model_2 = model_list[1].layers[0].kernel.value

        # Check if the new weights are different from the original weights
        assert not np.array_equal(weights_model_1, original_weights)
        assert not np.array_equal(weights_model_2, original_weights)

        # Check if the new weights are different from each other
        assert not np.array_equal(weights_model_1, weights_model_2)

    def test_parameters_are_independent(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if parameter values are independent between members."""
        model_list = ensemble(flax_model_small_2d_2d, num_members=2)
        a, b = model_list

        # clone the initial kernel of model B for later comparison
        kernel_b_before = b.layers[0].kernel.value.copy()

        # mutate model A's parameters
        a.layers[0].kernel.value = a.layers[0].kernel.value + 1.0

        # model B's parameters should remain unchanged
        kernel_b_after = b.layers[0].kernel.value

        assert np.array_equal(kernel_b_before, kernel_b_after)

    def test_invalid_num_members_type(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if providing a non-integer num_members raises a TypeError."""
        with pytest.raises(TypeError, match="num_members must be an int"):
            ensemble(flax_model_small_2d_2d, num_members="three")  # type: ignore[arg-type]

    def test_invalid_num_members_value(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if providing a negative num_members raises a ValueError."""
        with pytest.raises(ValueError, match="num_members must be non-negative"):
            ensemble(flax_model_small_2d_2d, num_members=-2)

    def test_two_ensembles_are_different(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if two ensembles created from the same base are different (wegen RNG)."""
        # create ensemble 1
        ensemble_1 = ensemble(flax_model_small_2d_2d, num_members=1)
        weights_e1 = ensemble_1[0].layers[0].kernel.value.copy()

        # create ensemble 2
        ensemble_2 = ensemble(flax_model_small_2d_2d, num_members=1)
        weights_e2 = ensemble_2[0].layers[0].kernel.value.copy()

        # weights should be different, because different RNGs should be used
        assert not np.array_equal(weights_e1, weights_e2), (
            "Error: RNG is not handled correctly, the two ensembles are identical!"
        )

    def test_no_reset_params_preserves_values(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Ensure that with reset_params=False, the parameter values are preserved."""
        # save original weights for comparison
        original_weights = flax_model_small_2d_2d.layers[0].kernel.value

        # create ensemble with reset_params=False
        # no weights should be changed
        model_list = ensemble(flax_model_small_2d_2d, num_members=2, reset_params=False)

        # weights should be identical to original
        weights_member_1 = model_list[0].layers[0].kernel.value
        weights_member_2 = model_list[1].layers[0].kernel.value

        assert np.array_equal(weights_member_1, original_weights), (
            "Weights should be identical to original when reset_params=False"
        )
        assert np.array_equal(weights_member_2, original_weights)

        # weights between members should also be identical
        assert np.array_equal(weights_member_1, weights_member_2)


class TestEnsembleFunctionality:
    """Tests for forward pass functionality."""

    @pytest.fixture
    def dummy_input(self) -> np.ndarray:
        """Creates a dummy input array."""
        return np.ones((5, 2), dtype=np.float32)

    def test_forward_passes_shape(
        self,
        flax_model_small_2d_2d: nnx.Sequential,
        dummy_input: np.ndarray,
    ) -> None:
        """Ensure that the ensemble forward pass produces outputs of expected shape."""
        n_members = 4

        model_list = ensemble(flax_model_small_2d_2d, num_members=n_members)

        # Collect outputs from each ensemble member
        # because of nnx.Sequential, we use __call__
        outputs = [model(dummy_input) for model in model_list]

        # Check that each output has the correct shape
        for output in outputs:
            assert output.shape == (5, 2)
            assert isinstance(output, jnp.ndarray)

    def test_forward_passes_output_difference(
        self,
        flax_model_small_2d_2d: nnx.Sequential,
        dummy_input: np.ndarray,
    ) -> None:
        """Ensure that different ensemble members produce different outputs for the same input."""
        n_members = 3

        model_list = ensemble(flax_model_small_2d_2d, num_members=n_members)

        # Collect outputs from each ensemble member
        outputs = [model(dummy_input) for model in model_list]

        # Check that outputs are not all the same
        output_1 = outputs[0]
        output_2 = outputs[1]

        # outputs should be different
        assert not np.array_equal(output_1, output_2), (
            "All ensemble members produced the same output, suggesting identical parameters!"
        )
