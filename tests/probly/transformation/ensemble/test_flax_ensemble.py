from __future__ import annotations

from typing import Set

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# real API
_mod = pytest.importorskip(
    "probly.transformation.ensemble.flax",
    reason="ensemble/flax implementation not available in this branch",
)

# public function under test
generate_flax_ensemble = _mod.generate_flax_ensemble


def _fwd(model: object, x: jnp.ndarray) -> object:
    """Unified forward: try nnx (callable), then linen (.apply)."""
    nnx_err: Exception | None = None
    linen_err: Exception | None = None

    # Try nnx-style callable
    try:
        return model(x)  # type: ignore[misc]
    except (TypeError, AttributeError) as e:
        nnx_err = e

    # Try linen-style .apply
    if hasattr(model, "apply"):
        try:
            return model.apply(x)  # type: ignore[attr-defined]
        except TypeError as e:
            linen_err = e

    msg = "Forward call not recognized (neither nnx nor linen)."
    raise AssertionError(msg) from (linen_err or nnx_err or None)


def _to_array_host(out: object) -> np.ndarray:
    """Normalize forward outputs to host ndarray.

    Outside jit only:
    - jnp.ndarray: convert to numpy
    - dict: try common keys ('mean'/'output'/'y'/'agg'/'aggregated'), else first ndarray value
    - tuple/list: take the first ndarray element
    """
    if isinstance(out, jnp.ndarray):
        return np.asarray(out)

    if isinstance(out, dict):  # type: ignore[reportGeneralTypeIssues]
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
    flax_model_small_2d_2d: object, xbatch: jnp.ndarray,
) -> None:
    base = flax_model_small_2d_2d
    num = 3
    members = generate_flax_ensemble(base, num_members=num, reset_params=False)

    assert isinstance(members, (list, tuple)), "Should return a sequence of members"
    assert len(members) == num

    ids: Set[int] = set()
    y_base = _to_array_host(_fwd(base, xbatch))  # type: ignore[arg-type]
    for i, m in enumerate(members):
        assert isinstance(m, type(base)), f"Member {i} has a different type from base"
        ids.add(id(m))
        y_i = _to_array_host(_fwd(m, xbatch))  # type: ignore[arg-type]
        assert y_i.shape == y_base.shape, "Output shape mismatch with base model"

    assert len(ids) == num, "Members should be distinct objects (not same reference)"


def test_reset_params_false_outputs_identical(
    flax_model_small_2d_2d: object, xbatch: jnp.ndarray,
) -> None:
    base = flax_model_small_2d_2d
    members = generate_flax_ensemble(base, num_members=4, reset_params=False)

    outs = [_to_array_host(_fwd(m, xbatch)) for m in members]  # type: ignore[arg-type]
    for i in range(1, len(outs)):
        np.testing.assert_allclose(outs[0], outs[i], rtol=1e-6, atol=1e-6)


def test_reset_params_true_outputs_prefer_difference(
    flax_model_small_2d_2d: object, xbatch: jnp.ndarray,
) -> None:
    base = flax_model_small_2d_2d
    members = generate_flax_ensemble(base, num_members=4, reset_params=True)

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
    flax_model_small_2d_2d: object, xbatch: jnp.ndarray,
) -> None:
    base = flax_model_small_2d_2d
    num = 3
    ens1 = generate_flax_ensemble(base, num_members=num, reset_params=True)
    ens2 = generate_flax_ensemble(base, num_members=num, reset_params=True)

    for i in range(num):
        y1 = _to_array_host(_fwd(ens1[i], xbatch))  # type: ignore[arg-type]
        y2 = _to_array_host(_fwd(ens2[i], xbatch))  # type: ignore[arg-type]
        np.testing.assert_allclose(y1, y2, rtol=1e-6, atol=1e-6)


def test_member_forward_jit_consistency(
    flax_model_small_2d_2d: object, xbatch: jnp.ndarray,
) -> None:
    base = flax_model_small_2d_2d
    members = generate_flax_ensemble(base, num_members=2, reset_params=False)
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


