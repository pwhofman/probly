"""Tests for flax HetNets transformation."""

from __future__ import annotations

from typing import cast

import pytest

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from probly.layers.flax import HeteroscedasticLayer  # noqa: E402
from probly.method.het_net import het_net  # noqa: E402
from tests.probly.flax_utils import count_layers  # noqa: E402


class TestHetNetLayerReplacement:
    """Tests for the structural changes het_net applies to a flax model."""

    def test_replaces_last_linear_with_heteroscedastic_layer(self, flax_model_small_2d_2d: nnx.Module) -> None:
        """Verify the final ``nnx.Linear`` is replaced and other linears are untouched."""
        original = flax_model_small_2d_2d
        original_linear_count = count_layers(original, nnx.Linear)

        new_model = cast("nnx.Module", het_net(original, num_factors=2, predictor_type="logit_classifier"))

        new_linear_count = count_layers(new_model, nnx.Linear)
        het_count = count_layers(new_model, HeteroscedasticLayer)

        assert het_count == 1
        # Each HeteroscedasticLayer contains 3 sub nnx.Linear instances (mu, diag, v).
        # The traversal therefore reports the original count - 1 + 3 = original + 2 linears.
        assert new_linear_count == original_linear_count + 2

    def test_returns_a_clone(self, flax_model_small_2d_2d: nnx.Module) -> None:
        """Ensure het_net does not mutate the input model."""
        new_model = het_net(flax_model_small_2d_2d, num_factors=2, predictor_type="logit_classifier")
        assert new_model is not flax_model_small_2d_2d


class TestHetNetForwardPass:
    """Tests that the transformed model produces logits of the expected shape."""

    def test_forward_pass_shape(self, flax_regression_model_2d: nnx.Module) -> None:
        """Verify the wrapped model returns ``(batch, num_classes)`` logits."""
        new_model = het_net(flax_regression_model_2d, num_factors=3, predictor_type="logit_classifier")
        out = new_model(jnp.ones((5, 4)))
        assert out.shape == (5, 2)

    def test_parameter_efficient_variant(self, flax_regression_model_2d: nnx.Module) -> None:
        """Verify the parameter-efficient routing also produces logits of the expected shape."""
        new_model = het_net(
            flax_regression_model_2d,
            num_factors=3,
            is_parameter_efficient=True,
            predictor_type="logit_classifier",
        )
        out = new_model(jnp.ones((5, 4)))
        assert out.shape == (5, 2)
