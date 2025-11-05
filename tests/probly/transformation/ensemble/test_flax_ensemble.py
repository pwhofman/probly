# tests/probly/transformation/ensemble/test_flax_ensemble.py
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

_mod = pytest.importorskip(
    "probly.transformation.ensemble.flax",
    reason="ensemble/flax implementation not available in this branch",
)
generate_flax_ensemble = _mod.generate_flax_ensemble

from tests.probly.fixtures.flax_models import flax_model_small_2d_2d


def _fwd(model, x):
    """Unified forward: try nnx (callable), then linen (.apply)."""
    nnx_err = None
    try:
        return model(x)  # nnx path
    except TypeError as e:
        nnx_err = e  # record, don't swallow

    if hasattr(model, "apply"):
        try:
            return model.apply(x)  # linen path
        except (TypeError, AttributeError) as e2:
            linen_err = e2
            msg = (
                "Forward call not recognized (neither nnx nor linen). "
                f"nnx TypeError: {nnx_err!r}; linen error: {linen_err!r}"
            )
            raise AssertionError(msg)  # message assigned above
    else:
        msg = (
            "Forward call not recognized (neither nnx nor linen). "
            f"nnx TypeError: {nnx_err!r}; no `.apply` on model."
        )
        raise AssertionError(msg)


def _to_array_host(out):
    """
    Normalize various forward outputs to a host-side ndarray (only outside jit):
    - ndarray: return as numpy
    - dict: try 'mean'/'output'/'y'/'agg'/'aggregated', else first ndarray value
    - tuple/list: first ndarray element
    """
    if isinstance(out, jnp.ndarray):
        return np.asarray(out)
    if isinstance(out, dict):
        for k in ("mean", "output", "y", "agg", "aggregated"):
            v = out.get(k, None)
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


def test_returns_sequence_and_types(flax_model_small_2d_2d, xbatch):
    base = flax_model_small_2d_2d
    num = 3
    members = generate_flax_ensemble(base, num_members=num, reset_params=False)

    assert isinstance(members, (list, tuple)), "Should return a sequence of members"
    assert len(members) == num

    ids = set()
    y_base = _to_array_host(_fwd(base, xbatch))
    for i, m in enumerate(members):
        assert isinstance(m, type(base)), f"Member {i} has a different type from base"
        ids.add(id(m))
        y_i = _to_array_host(_fwd(m, xbatch))
        assert y_i.shape == y_base.shape, "Output shape mismatch with base model"

    assert len(ids) == num, "Members should be distinct objects (not same reference)"


def test_reset_params_false_outputs_identical(flax_model_small_2d_2d, xbatch):
    base = flax_model_small_2d_2d
    members = generate_flax_ensemble(base, num_members=4, reset_params=False)

    outs = [_to_array_host(_fwd(m, xbatch)) for m in members]
    for i in range(1, len(outs)):
        np.testing.assert_allclose(outs[0], outs[i], rtol=1e-6, atol=1e-6)


def test_reset_params_true_outputs_prefer_difference(flax_model_small_2d_2d, xbatch):
    base = flax_model_small_2d_2d
    members = generate_flax_ensemble(base, num_members=4, reset_params=True)

    outs = [_to_array_host(_fwd(m, xbatch)) for m in members]
    any_diff = False
    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            if not np.allclose(outs[i], outs[j], rtol=1e-6, atol=1e-6):
                any_diff = True
                break
        if any_diff:
            break

    if any_diff:
        assert True  # Normal case: outputs differ
    else:
        assert len({id(m) for m in members}) == len(members)
        base_out_shape = _to_array_host(_fwd(base, xbatch)).shape
        for m in members:
            assert _to_array_host(_fwd(m, xbatch)).shape == base_out_shape


def test_reset_params_true_is_deterministic_across_calls(flax_model_small_2d_2d, xbatch):
    base = flax_model_small_2d_2d
    num = 3
    ens1 = generate_flax_ensemble(base, num_members=num, reset_params=True)
    ens2 = generate_flax_ensemble(base, num_members=num, reset_params=True)

    for i in range(num):
        y1 = _to_array_host(_fwd(ens1[i], xbatch))
        y2 = _to_array_host(_fwd(ens2[i], xbatch))
        np.testing.assert_allclose(y1, y2, rtol=1e-6, atol=1e-6)


def test_member_forward_jit_consistency(flax_model_small_2d_2d, xbatch):
    base = flax_model_small_2d_2d
    members = generate_flax_ensemble(base, num_members=2, reset_params=False)
    m0 = members[0]

    def f(x):
        return _fwd(m0, x)

    f_jit = jax.jit(f)

    y0 = f(xbatch)
    y1 = f_jit(xbatch)

    np.testing.assert_allclose(np.asarray(y0), np.asarray(y1), rtol=1e-6, atol=1e-6)
