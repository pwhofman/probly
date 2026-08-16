"""Test for flax bayesian models."""

from __future__ import annotations

from typing import cast

from flax import nnx
import pytest

from probly.layers.flax import BayesLinear
from probly.transformation.bayesian import bayesian
from tests.probly.flax_utils import count_layers


@pytest.fixture
def flax_model_small_2d_2d(flax_rngs: nnx.Rngs) -> nnx.Module:
    """Return a small linear model with 2 input and 2 output neurons."""
    model = nnx.Sequential(
        nnx.Linear(2, 2, rngs=flax_rngs),
        nnx.Linear(2, 2, rngs=flax_rngs),
        nnx.Linear(2, 2, rngs=flax_rngs),
    )
    return model


class TestNetworkArchitecture:
    def test_replace_linear_layer(self, flax_model_small_2d_2d: nnx.Module) -> None:
        model = cast("nnx.Module", bayesian(flax_model_small_2d_2d))

        count_linear_original = count_layers(flax_model_small_2d_2d, nnx.Linear)
        count_bayes_modified = count_layers(model, BayesLinear)
        count_linear_modified = count_layers(model, nnx.Linear)

        assert isinstance(model, type(flax_model_small_2d_2d))
        assert count_linear_modified == 0
        assert count_bayes_modified == count_linear_original
