"""Tests for the flax normal-inverse-gamma head transformation."""

from __future__ import annotations

import pytest

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from probly.layers.flax import NormalInverseGammaLinear  # noqa: E402
from probly.transformation.normal_inverse_gamma_head import normal_inverse_gamma_head  # noqa: E402
from tests.probly.flax_utils import count_layers  # noqa: E402


def test_returns_a_clone(flax_regression_model_2d: nnx.Module) -> None:
    """normal_inverse_gamma_head returns a clone, leaving the original untouched."""
    new_model = normal_inverse_gamma_head(flax_regression_model_2d)

    assert new_model is not flax_regression_model_2d


def test_replaces_only_last_linear_layer(flax_regression_model_2d: nnx.Module) -> None:
    """Exactly one ``NormalInverseGammaLinear`` is inserted, and one ``nnx.Linear`` removed."""
    original = flax_regression_model_2d
    original_linear_count = count_layers(original, nnx.Linear)

    new_model = normal_inverse_gamma_head(original)
    new_linear_count = count_layers(new_model, nnx.Linear)
    nig_count = count_layers(new_model, NormalInverseGammaLinear)

    assert nig_count == 1
    assert new_linear_count == original_linear_count - 1


def test_forward_pass_returns_nig_dict(flax_regression_model_2d: nnx.Module) -> None:
    """The transformed model returns the four NIG parameter heads with consistent shapes."""
    new_model = normal_inverse_gamma_head(flax_regression_model_2d)
    out = new_model(jnp.ones((3, 4)))

    assert isinstance(out, dict)
    assert set(out.keys()) == {"gamma", "nu", "alpha", "beta"}

    expected_shape = (3, 2)
    for value in out.values():
        assert value.shape == expected_shape

    # nu, beta are softplus → non-negative; alpha is softplus + 1 → at least one
    assert (out["nu"] >= 0).all()
    assert (out["beta"] >= 0).all()
    assert (out["alpha"] >= 1).all()
