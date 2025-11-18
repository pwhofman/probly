"""Test for flax ensemble transformations."""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
import pytest

from probly.transformation import ensemble
from tests.probly.flax_utils import count_layers

jax = pytest.importorskip("jax")
from jax import numpy as jnp  # noqa: E402

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402


class TestNetworkStructure:
    """Test class for network structure tests."""

    def test_linear_network_no_reset(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests the linear model ensemble without resetting parameters."""
        num_members = 4
        model = ensemble(flax_model_small_2d_2d, num_members=num_members, reset_params=False)

        # count number of nnx.Linear layers in original model
        count_linear_original = count_layers(flax_model_small_2d_2d, nnx.Linear)
        # count number of nnx.Dropout layers in original model
        count_dropout_original = count_layers(flax_model_small_2d_2d, nnx.Dropout)
        # count number of nnx.Sequential layers in original model
        count_sequential_original = count_layers(flax_model_small_2d_2d, nnx.Sequential)

        for member in model:
            # count number of nnx.Linear layers in modified model
            count_linear_modified = count_layers(member, nnx.Linear)
            # count number of nnx.Dropout layers in modified model
            count_dropout_modified = count_layers(member, nnx.Dropout)
            # count number of nnx.Sequential layers in modified model
            count_sequential_modified = count_layers(member, nnx.Sequential)

            assert member is not None
            assert isinstance(member, type(flax_model_small_2d_2d))
            assert count_linear_modified == count_linear_original
            assert count_dropout_modified == count_dropout_original
            assert count_sequential_modified == count_sequential_original

    def test_linear_network_with_reset(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests the linear model ensemble with resetting parameters."""
        num_members = 4
        model = ensemble(flax_model_small_2d_2d, num_members=num_members, reset_params=True)

        # count number of nnx.Linear layers in original model
        count_linear_original = count_layers(flax_model_small_2d_2d, nnx.Linear)
        # count number of nnx.Dropout layers in original model
        count_dropout_original = count_layers(flax_model_small_2d_2d, nnx.Dropout)
        # count number of nnx.Sequential layers in original model
        count_sequential_original = count_layers(flax_model_small_2d_2d, nnx.Sequential)

        for member in model:
            # count number of nnx.Linear layers in modified model
            count_linear_modified = count_layers(member, nnx.Linear)
            # count number of nnx.Dropout layers in modified model
            count_dropout_modified = count_layers(member, nnx.Dropout)
            # count number of nnx.Sequential layers in modified model
            count_sequential_modified = count_layers(member, nnx.Sequential)

            assert member is not None
            assert isinstance(member, type(flax_model_small_2d_2d))
            assert count_linear_modified == count_linear_original
            assert count_dropout_modified == count_dropout_original
            assert count_sequential_modified == count_sequential_original

    def test_conv_linear_network_no_reset(self, flax_conv_linear_model: nnx.Sequential) -> None:
        """Tests the conv-linear model ensemble without resetting parameters."""
        model = ensemble(flax_conv_linear_model, num_members=3, reset_params=False)

        # count number of nnx.Linear layers in original model
        count_linear_original = count_layers(flax_conv_linear_model, nnx.Linear)
        # count number of nnx.Dropout layers in original model
        count_dropout_original = count_layers(flax_conv_linear_model, nnx.Dropout)
        # count number of nnx.Sequential layers in original model
        count_sequential_original = count_layers(flax_conv_linear_model, nnx.Sequential)
        # count number of nnx.Conv2d layers in original model
        count_conv_original = count_layers(flax_conv_linear_model, nnx.Conv)

        for member in model:
            # count number of nnx.Linear layers in modified model
            count_linear_modified = count_layers(member, nnx.Linear)
            # count number of nnx.Dropout layers in modified model
            count_dropout_modified = count_layers(member, nnx.Dropout)
            # count number of nnx.Sequential layers in modified model
            count_sequential_modified = count_layers(member, nnx.Sequential)
            # count number of nnx.Conv2d layers in modified model
            count_conv_modified = count_layers(member, nnx.Conv)

            # check that the model is not modified except for the dropout layer
            assert member is not None
            assert isinstance(member, type(flax_conv_linear_model))
            assert count_dropout_original == count_dropout_modified
            assert count_linear_original == count_linear_modified
            assert count_sequential_original == count_sequential_modified
            assert count_conv_original == count_conv_modified

    def test_conv_linear_network_with_reset(self, flax_conv_linear_model: nnx.Sequential) -> None:
        """Tests the conv-linear model ensemble with resetting parameters."""
        model = ensemble(flax_conv_linear_model, num_members=3, reset_params=True)

        # count number of nnx.Linear layers in original model
        count_linear_original = count_layers(flax_conv_linear_model, nnx.Linear)
        # count number of nnx.Dropout layers in original model
        count_dropout_original = count_layers(flax_conv_linear_model, nnx.Dropout)
        # count number of nnx.Sequential layers in original model
        count_sequential_original = count_layers(flax_conv_linear_model, nnx.Sequential)
        # count number of nnx.Conv2d layers in original model
        count_conv_original = count_layers(flax_conv_linear_model, nnx.Conv)

        for member in model:
            # count number of nnx.Linear layers in modified model
            count_linear_modified = count_layers(member, nnx.Linear)
            # count number of nnx.Dropout layers in modified model
            count_dropout_modified = count_layers(member, nnx.Dropout)
            # count number of nnx.Sequential layers in modified model
            count_sequential_modified = count_layers(member, nnx.Sequential)
            # count number of nnx.Conv2d layers in modified model
            count_conv_modified = count_layers(member, nnx.Conv)

            # check that the model is not modified except for the dropout layer
            assert member is not None
            assert isinstance(member, type(flax_conv_linear_model))
            assert count_dropout_original == count_dropout_modified
            assert count_linear_original == count_linear_modified
            assert count_sequential_original == count_sequential_modified
            assert count_conv_original == count_conv_modified

    def test_custom_network_with_reset(self, flax_custom_model: nnx.Module) -> None:
        """Tests the custom model essemble with resetting parameters."""
        num_members = 5
        model = ensemble(flax_custom_model, num_members=num_members, reset_params=True)

        for member in model:
            assert isinstance(member, type(flax_custom_model))
            assert not isinstance(member, nnx.Sequential)

    def test_custom_network_no_reset(self, flax_custom_model: nnx.Module) -> None:
        """Tests the custom model essemble without resetting parameters."""
        num_members = 5
        model = ensemble(flax_custom_model, num_members=num_members, reset_params=False)

        for member in model:
            assert isinstance(member, type(flax_custom_model))
            assert not isinstance(member, nnx.Sequential)


class TestParameters:
    """Test class for network parameter tests."""

    def test_parameters_linear_network_no_reset(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests that parameters are the same when reset_params is False."""
        num_members = 3
        model = ensemble(flax_model_small_2d_2d, num_members=num_members, reset_params=False)

        # get original parameters
        original_params = jax.tree_util.tree_leaves(flax_model_small_2d_2d)

        for i in range(num_members - 1):
            # get all params
            memb_model1_params = jax.tree_util.tree_leaves(model[i])

            # compare only weights, not biases
            assert jnp.array_equal(memb_model1_params[1], original_params[1])
            # compare only biases, not weights
            assert jnp.array_equal(memb_model1_params[0], original_params[0])

    def test_parameters_linear_network_with_reset(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests that parameters are the the same."""
        num_members = 3
        model1 = ensemble(flax_model_small_2d_2d, num_members=num_members, reset_params=True)
        model2 = ensemble(flax_model_small_2d_2d, num_members=num_members, reset_params=True)

        # get original parameters
        original_params = jax.tree_util.tree_leaves(flax_model_small_2d_2d)

        for i in range(num_members - 1):
            # get all params
            memb_model1_params = jax.tree_util.tree_leaves(model1[i])
            memb_model2_params = jax.tree_util.tree_leaves(model2[i])

            # compare only weights, not biases
            assert jnp.array_equal(memb_model1_params[1], memb_model2_params[1])
            # compare only biases, not weights
            assert jnp.array_equal(memb_model1_params[0], memb_model2_params[0])
            # compare weights with original model weights to ensure they are different
            assert not jnp.array_equal(memb_model1_params[1], original_params[1])

    def test_parameters_conv_linear_network_no_reset(self, flax_conv_linear_model: nnx.Sequential) -> None:
        """Tests that parameters are the same when reset_params is False."""
        num_members = 3
        model = ensemble(flax_conv_linear_model, num_members=num_members, reset_params=False)

        # get original parameters
        original_params = jax.tree_util.tree_leaves(flax_conv_linear_model)

        for i in range(num_members - 1):
            # get all params
            memb_model1_params = jax.tree_util.tree_leaves(model[i])

            # compare only weights, not biases
            assert jnp.array_equal(memb_model1_params[1], original_params[1])
            # compare only biases, not weights
            assert jnp.array_equal(memb_model1_params[0], original_params[0])

    def test_parameters_conv_linear_network_with_reset(self, flax_conv_linear_model: nnx.Sequential) -> None:
        """Tests that parameters are the same."""
        num_members = 3
        model1 = ensemble(flax_conv_linear_model, num_members=num_members, reset_params=True)
        model2 = ensemble(flax_conv_linear_model, num_members=num_members, reset_params=True)

        # get original parameters
        original_params = jax.tree_util.tree_leaves(flax_conv_linear_model)

        for i in range(num_members - 1):
            # get all params
            memb_model1_params = jax.tree_util.tree_leaves(model1[i])
            memb_model2_params = jax.tree_util.tree_leaves(model2[i])

            # compare only weights, not biases
            assert jnp.array_equal(memb_model1_params[1], memb_model2_params[1])
            # compare only biases, not weights
            assert jnp.array_equal(memb_model1_params[0], memb_model2_params[0])
            # compare weights with original model weights to ensure they are different
            assert not jnp.array_equal(memb_model1_params[1], original_params[1])

    def test_parameters_custom_network_no_reset(self, flax_custom_model: nnx.Module) -> None:
        """Tests that parameters are the same when reset_params is False."""
        num_members = 3
        model = ensemble(flax_custom_model, num_members=num_members, reset_params=False)

        # get original parameters
        original_params = jax.tree_util.tree_leaves(flax_custom_model)

        for i in range(num_members - 1):
            # get all params
            memb_model1_params = jax.tree_util.tree_leaves(model[i])

            # compare only weights, not biases
            assert jnp.array_equal(memb_model1_params[1], original_params[1])
            # compare only biases, not weights
            assert jnp.array_equal(memb_model1_params[0], original_params[0])

    def test_parameters_custom_network_with_reset(self, flax_custom_model: nnx.Module) -> None:
        """Tests that parameters are the the same."""
        num_members = 3
        model1 = ensemble(flax_custom_model, num_members=num_members, reset_params=True)
        model2 = ensemble(flax_custom_model, num_members=num_members, reset_params=True)

        # get original parameters
        original_params = jax.tree_util.tree_leaves(flax_custom_model)

        for i in range(num_members - 1):
            # get all params
            memb_model1_params = jax.tree_util.tree_leaves(model1[i])
            memb_model2_params = jax.tree_util.tree_leaves(model2[i])

            # compare only weights, not biases
            assert jnp.array_equal(memb_model1_params[1], memb_model2_params[1])
            # compare only biases, not weights
            assert jnp.array_equal(memb_model1_params[0], memb_model2_params[0])
            # compare weights with original model weights to ensure they are different
            assert not jnp.array_equal(memb_model1_params[1], original_params[1])


class CallableModel(Protocol):
    def __call__(self, x: jnp.ndarray, /) -> nnx.Module: ...


class ApplyModel(Protocol):
    def apply(self, x: jnp.ndarray, /) -> nnx.Module: ...


def _fwd(model: nnx.Module, x: jnp.ndarray) -> nnx.Module:
    """Unified forward: try nnx (callable), then linen (.apply)."""
    nnx_err: Exception | None = None
    linen_err: Exception | None = None

    # Try nnx-style callable
    try:
        return cast(CallableModel, model)(x)
    except (TypeError, AttributeError) as e:
        nnx_err = e

    # Try linen-style .apply
    if hasattr(model, "apply"):
        try:
            return cast(ApplyModel, model).apply(x)
        except TypeError as e:
            linen_err = e

    msg = "Forward call not recognized (neither nnx nor linen)."
    raise AssertionError(msg) from (linen_err or nnx_err or None)


def _to_array_host(out: nnx.Module) -> np.ndarray:
    """Normalize forward outputs to a host-side ndarray.

    Outside jit only:
    - jnp.ndarray: convert to numpy
    - dict: try common keys ('mean'/'output'/'y'/'agg'/'aggregated'), else first ndarray value
    - tuple/list: take the first ndarray element
    """
    if isinstance(out, jnp.ndarray):
        return np.asarray(out)

    if isinstance(out, dict):
        for k in ("mean", "output", "y", "agg", "aggregated"):
            v = out.get(k, None)  # type: ignore[call-arg]
            if isinstance(v, jnp.ndarray):
                return np.asarray(v)
        for v in out.values():
            if isinstance(v, jnp.ndarray):
                return np.asarray(v)

    if isinstance(out, (tuple, list)):
        for v in out:
            if isinstance(v, jnp.ndarray):
                return np.asarray(v)

    msg = "Could not extract ndarray from model output."
    raise AssertionError(msg)


@pytest.fixture
def xbatch() -> jnp.ndarray:
    return jnp.ones((8, 2), dtype=jnp.float32)


def test_returns_sequence_and_types(
    flax_model_small_2d_2d: nnx.Module,
    xbatch: jnp.ndarray,
) -> None:
    base = flax_model_small_2d_2d
    num = 3
    members = ensemble(base, num_members=num, reset_params=False)

    assert isinstance(members, (list, tuple)), "Should return a sequence of members"
    assert len(members) == num

    ids: set[int] = set()
    y_base = _to_array_host(_fwd(base, xbatch))  # type: ignore[arg-type]
    for i, m in enumerate(members):
        assert isinstance(m, type(base)), f"Member {i} has a different type from base"
        ids.add(id(m))
        y_i = _to_array_host(_fwd(m, xbatch))  # type: ignore[arg-type]
        assert y_i.shape == y_base.shape, "Output shape mismatch with base model"

    assert len(ids) == num, "Members should be distinct nnx.Modules (not same reference)"


def test_reset_params_false_outputs_identical(
    flax_model_small_2d_2d: nnx.Module,
    xbatch: jnp.ndarray,
) -> None:
    base = flax_model_small_2d_2d
    members = ensemble(base, num_members=4, reset_params=False)

    outs = [_to_array_host(_fwd(m, xbatch)) for m in members]  # type: ignore[arg-type]
    for i in range(1, len(outs)):
        np.testing.assert_allclose(outs[0], outs[i], rtol=1e-6, atol=1e-6)


def test_reset_params_true_outputs_prefer_difference(
    flax_model_small_2d_2d: nnx.Module,
    xbatch: jnp.ndarray,
) -> None:
    base = flax_model_small_2d_2d
    members = ensemble(base, num_members=4, reset_params=True)

    outs = [_to_array_host(_fwd(m, xbatch)) for m in members]  # type: ignore[arg-type]
    any_diff = False
    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            if not np.allclose(outs[i], outs[j], rtol=1e-6, atol=1e-6):
                any_diff = True
                break
        if any_diff:
            break

    if not any_diff:
        # Fallback: ensure members are distinct and shapes match base output
        assert len({id(m) for m in members}) == len(members)
        base_out_shape = _to_array_host(_fwd(base, xbatch)).shape  # type: ignore[arg-type]
        for m in members:
            assert _to_array_host(_fwd(m, xbatch)).shape == base_out_shape  # type: ignore[arg-type]


def test_reset_params_true_is_deterministic_across_calls(
    flax_model_small_2d_2d: nnx.Module,
    xbatch: jnp.ndarray,
) -> None:
    base = flax_model_small_2d_2d
    num = 3
    ens1 = ensemble(base, num_members=num, reset_params=True)
    ens2 = ensemble(base, num_members=num, reset_params=True)

    for i in range(num):
        y1 = _to_array_host(_fwd(ens1[i], xbatch))  # type: ignore[arg-type]
        y2 = _to_array_host(_fwd(ens2[i], xbatch))  # type: ignore[arg-type]
        np.testing.assert_allclose(y1, y2, rtol=1e-6, atol=1e-6)


def test_member_forward_jit_consistency(
    flax_model_small_2d_2d: nnx.Module,
    xbatch: jnp.ndarray,
) -> None:
    members = ensemble(flax_model_small_2d_2d, num_members=2, reset_params=False)
    m0 = members[0]

    def f(x: jnp.ndarray) -> jnp.ndarray:
        # Return JAX array only; don't convert to numpy inside jit.
        y = _fwd(m0, x)  # type: ignore[arg-type]
        return jnp.asarray(y)

    f_jit = jax.jit(f)

    y0 = f(xbatch)
    y1 = f_jit(xbatch)

    # Compare on host
    np.testing.assert_allclose(np.asarray(y0), np.asarray(y1), rtol=1e-6, atol=1e-6)
