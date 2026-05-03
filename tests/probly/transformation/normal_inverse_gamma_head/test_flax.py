"""Tests for the flax normal-inverse-gamma head transformation."""

from __future__ import annotations

from typing import cast

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


def test_distinct_rngs_yield_distinct_layers(flax_regression_model_2d: nnx.Module) -> None:
    """Different ``rngs`` seeds produce distinct ``NormalInverseGammaLinear`` parameters."""
    m1 = normal_inverse_gamma_head(flax_regression_model_2d, rngs=nnx.Rngs(0))
    m2 = normal_inverse_gamma_head(flax_regression_model_2d, rngs=nnx.Rngs(42))

    seq1 = cast("nnx.Sequential", m1)
    seq2 = cast("nnx.Sequential", m2)
    nig1 = next(layer for layer in seq1.layers if isinstance(layer, NormalInverseGammaLinear))
    nig2 = next(layer for layer in seq2.layers if isinstance(layer, NormalInverseGammaLinear))

    assert not jnp.allclose(nig1.gamma.value, nig2.gamma.value)


def test_earlier_layers_unchanged(flax_regression_model_2d: nnx.Module) -> None:
    """Layers before the replaced final ``Linear`` retain their original types."""
    original = cast("nnx.Sequential", flax_regression_model_2d)
    new_model = cast("nnx.Sequential", normal_inverse_gamma_head(flax_regression_model_2d))

    last_layer_index = len(original.layers) - 1
    for i in range(last_layer_index):
        assert type(original.layers[i]) is type(new_model.layers[i])

    assert isinstance(new_model.layers[last_layer_index], NormalInverseGammaLinear)
