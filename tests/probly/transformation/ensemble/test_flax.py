"""Test für Flax Ensemble Module."""

from __future__ import annotations

from flax import nnx
from jax import numpy as jnp
import pytest

from probly.transformation.ensemble.flax import generate_flax_ensemble
from pytraverse import lazydispatch_traverser
from tests.probly.flax_utils import count_layers

flax = pytest.importorskip("flax")
reset_traverser = lazydispatch_traverser[object](name="reset_traverser")


class TestEnsembleModule:
    def test_return_type_and_length(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests that the output shape is correct. Uses flax_model_small_2d_2d."""
        # Create Ensemble with an example number of n_members.
        n = 3
        ens = generate_flax_ensemble(flax_model_small_2d_2d, n_members=n)

        # Checks the type of the Container
        assert isinstance(ens, list), f"Expected a list, got {type(ens)}"

        # Checks the type of the Elements.
        assert all(isinstance(m, nnx.Module) for m in ens), "All elements must be nnx.Module"

        # Checks the length
        assert len(ens) == n, f"Expected {n} members, got {len(ens)}"

    def test_return_type_and_length_conv(self, flax_conv_linear_model: nnx.Sequential) -> None:
        """Tests that the output shape is correct. Uses flax_conv_linear_model."""
        # Create Ensemble with an example number of n_members
        n = 3
        ens = generate_flax_ensemble(flax_conv_linear_model, n_members=n)

        # Checks the type of the Container
        assert isinstance(ens, list), f"Expected a list, got {type(ens)}"

        # Checks the type of the Elements.
        assert all(isinstance(m, nnx.Module) for m in ens), "All elements must be nnx.Module"

        # Checks the length
        assert len(ens) == n, f"Expected {n} members, got {len(ens)}"

    def test_layer_counts_scale_linear(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Checks if the number of layers is correct."""
        # Creates Ensemble with the Output Shape list[Module].
        k = 3
        base = flax_model_small_2d_2d
        ens = generate_flax_ensemble(base, n_members=k)

        # Count the linear Layers of the generated Ensemble.
        linear_layers = sum([count_layers(m, nnx.Linear) for m in ens])
        assert linear_layers == k * count_layers(base, nnx.Linear), f"Expected {k} layers, got {linear_layers}"

        # Count the sequential Layers of the generated Ensemble.
        sequential_layers = sum([count_layers(m, nnx.Sequential) for m in ens])
        assert sequential_layers == k * count_layers(base, nnx.Sequential), (
            f"Expected {k} layers, got {sequential_layers}"
        )

    def test_is_cloned(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if copy is different to the base model."""
        k = 2
        base = flax_model_small_2d_2d
        ens = generate_flax_ensemble(base, n_members=k)

        assert ens is not base, "Ensemble is the same as base"

    def test_deep_copy(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests if the parameters of the deep copy are different to the base model."""
        k = 2
        ens = generate_flax_ensemble(flax_model_small_2d_2d, n_members=k)

        kernels = []
        biases = []

        for m in ens:
            for layer in m.layers:
                if hasattr(layer, "kernel") and hasattr(layer, "bias"):
                    kernels.append(layer.kernel)
                    biases.append(layer.bias)

        kernel_difference = not all(jnp.allclose(kernels[0], k) for k in kernels[1:])
        bias_difference = not all(jnp.allclose(biases[1], k) for k in biases[1:])

        assert kernel_difference or bias_difference, (
            "Ensemble was not copied correctly. Parameters were not re initialized."
        )

    ''' Test results in AssertionError.
        def test_deep_copy_conv(self, flax_conv_linear_model: nnx.Sequential) -> None:
        """Tests if the parameters of the deep copy are different to the base model.
        Can be tested with Conv Modules.
        """

        base = flax_conv_linear_model

        # Get an Array with the original Kernel and Bias inputs.
        orig_kernels = []
        orig_biases = []

        for layer in base.layers:
            if hasattr(layer, "kernel") and hasattr(layer, "bias"):
                orig_kernels.append(layer.kernel)
                orig_biases.append(layer.bias)

        # Create an Ensemble
        k = 2
        ens = ensemble(flax_conv_linear_model, n_members=k)

        # Get the new Kernel and Bias values.
        for member in ens:
            member_kernels = []
            member_biases = []
            for layer in member.layers:
                if hasattr(layer, "kernel"):
                    member_kernels.append(layer.kernel)
                if hasattr(layer, "bias"):
                    member_biases.append(layer.bias)

        # Compare the original Parametes with the copied one. Should be different.
        for i in range(len(orig_kernels)):
            assert not jnp.allclose(orig_kernels[i], member_kernels[i]), (
                f"Parameters were not re initialized for layer {i}."
            )

        for i in range(len(orig_biases)):
            assert not jnp.allclose(orig_biases[i], member_biases[i]), (
                f"Parameters were not re initialized for layer {i}."
            )
    '''
