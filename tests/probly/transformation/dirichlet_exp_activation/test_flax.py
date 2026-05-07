"""Tests for the flax dirichlet exp-activation transformation."""

from __future__ import annotations

import pytest

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from probly.predictor import predict_raw  # noqa: E402
from probly.transformation.dirichlet_exp_activation import dirichlet_exp_activation  # noqa: E402


def test_wraps_model_in_sequential_with_exp(flax_model_small_2d_2d: nnx.Module) -> None:
    """The wrapped model is a Sequential whose final layer applies ``exp``."""
    wrapped = dirichlet_exp_activation(flax_model_small_2d_2d, predictor_type="logit_classifier")

    assert isinstance(wrapped, nnx.Sequential)
    # The wrapped model should be original ``Sequential`` followed by an ``_Exp`` module.
    assert len(wrapped.layers) == 2


def test_outputs_are_non_negative(flax_model_small_2d_2d: nnx.Module) -> None:
    """``exp`` always produces non-negative outputs, suitable as Dirichlet alpha."""
    wrapped = dirichlet_exp_activation(flax_model_small_2d_2d, predictor_type="logit_classifier")
    out = predict_raw(wrapped, jnp.ones((3, 2)))

    assert out.shape == (3, 2)
    assert (out >= 0).all()


def test_exp_relationship_holds(flax_model_small_2d_2d: nnx.Module) -> None:
    """Ensure the wrapped output is ``exp`` of the original logits."""
    wrapped = dirichlet_exp_activation(flax_model_small_2d_2d, predictor_type="logit_classifier")
    x = jnp.ones((3, 2))
    raw_logits = flax_model_small_2d_2d(x)
    wrapped_out = predict_raw(wrapped, x)

    assert jnp.allclose(wrapped_out, jnp.exp(raw_logits), atol=1e-5)
