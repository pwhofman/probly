"""Tests for the flax dirichlet softplus-activation transformation."""

from __future__ import annotations

import pytest

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402
import jax.nn  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from probly.predictor import predict_raw  # noqa: E402
from probly.transformation.dirichlet_softplus_activation import dirichlet_softplus_activation  # noqa: E402


def test_wraps_model_in_sequential_with_softplus_and_add_one(flax_model_small_2d_2d: nnx.Module) -> None:
    """The wrapped model is a Sequential whose tail applies ``softplus`` then ``+1``."""
    wrapped = dirichlet_softplus_activation(flax_model_small_2d_2d, predictor_type="logit_classifier")

    assert isinstance(wrapped, nnx.Sequential)
    # original Sequential + Softplus + AddOne
    assert len(wrapped.layers) == 3


def test_outputs_are_at_least_one(flax_model_small_2d_2d: nnx.Module) -> None:
    """``softplus(x) + 1 >= 1`` always — required for valid Dirichlet alpha."""
    wrapped = dirichlet_softplus_activation(flax_model_small_2d_2d, predictor_type="logit_classifier")
    out = predict_raw(wrapped, jnp.ones((3, 2)))

    assert out.shape == (3, 2)
    assert (out >= 1).all()


def test_softplus_relationship_holds(flax_model_small_2d_2d: nnx.Module) -> None:
    """Ensure the wrapped output equals ``softplus(logits) + 1``."""
    wrapped = dirichlet_softplus_activation(flax_model_small_2d_2d, predictor_type="logit_classifier")
    x = jnp.ones((3, 2))
    raw_logits = flax_model_small_2d_2d(x)
    wrapped_out = predict_raw(wrapped, x)

    assert jnp.allclose(wrapped_out, jax.nn.softplus(raw_logits) + 1, atol=1e-5)
