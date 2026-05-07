"""Tests for the flax class-bias ensemble transformation."""

from __future__ import annotations

from typing import cast

import pytest

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from probly.transformation.class_bias_ensemble import class_bias_ensemble  # noqa: E402


def _last_linear_bias(model: nnx.Module) -> jnp.ndarray:
    """Return the bias array of the last ``nnx.Linear`` in ``model``."""
    seq = cast("nnx.Sequential", model)
    last = cast("nnx.Linear", seq.layers[-1])
    assert last.bias is not None
    return last.bias.value


def test_returns_num_members(flax_model_small_2d_2d: nnx.Module) -> None:
    """class_bias_ensemble returns exactly ``num_members`` cloned models."""
    members = class_bias_ensemble(
        flax_model_small_2d_2d,
        num_members=4,
        reset_params=False,
        predictor_type="logit_classifier",
    )

    assert len(members) == 4


def test_first_member_keeps_bias_unchanged(flax_model_small_2d_2d: nnx.Module) -> None:
    """The first ensemble member (``BIAS_CLS=0``) leaves the final bias untouched."""
    members = class_bias_ensemble(
        flax_model_small_2d_2d,
        num_members=2,
        reset_params=False,
        predictor_type="logit_classifier",
    )

    assert jnp.array_equal(_last_linear_bias(members[0]), _last_linear_bias(flax_model_small_2d_2d))


def test_subsequent_members_set_per_class_bias(flax_model_small_2d_2d: nnx.Module) -> None:
    """Members ``i > 0`` set bias index ``(i-1) % out_features`` to ``tobias_value``."""
    seq = cast("nnx.Sequential", flax_model_small_2d_2d)
    out_features = cast("nnx.Linear", seq.layers[-1]).out_features
    members = class_bias_ensemble(
        flax_model_small_2d_2d,
        num_members=out_features + 1,
        reset_params=False,
        tobias_value=42,
        predictor_type="logit_classifier",
    )

    for i in range(1, out_features + 1):
        bias = _last_linear_bias(members[i])
        idx = (i - 1) % out_features
        assert bias[idx] == 42.0


def test_reset_params_true_yields_distinct_kernels(flax_model_small_2d_2d: nnx.Module) -> None:
    """When ``reset_params=True``, each member gets independently reinitialized weights."""
    members = class_bias_ensemble(
        flax_model_small_2d_2d,
        num_members=3,
        reset_params=True,
        rngs=nnx.Rngs(0),
        predictor_type="logit_classifier",
    )

    kernels = [m.layers[0].kernel.value for m in members]
    assert not jnp.allclose(kernels[0], kernels[1])
    assert not jnp.allclose(kernels[1], kernels[2])
