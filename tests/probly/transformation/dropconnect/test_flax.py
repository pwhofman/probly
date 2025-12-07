"""Tests for the NNX DropConnect transformation registration and replacement."""

from __future__ import annotations

import importlib
from typing import Any, cast

import pytest

from probly.layers.flax import Dense, DropConnectDense
from probly.transformation.dropconnect import common as c, flax as flax_mod

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402

jax = pytest.importorskip("jax")
from jax import numpy as jnp  # noqa: E402


def test_nnx_register_calls_dropconnect_traverser_on_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Import-time register should wire Dense with replace_nnx_dropconnect_dense and pass P."""
    called: dict[str, Any] = {}

    def fake_register(*, cls: type[Any], traverser: Any, **kwargs: Any) -> None:  # noqa: ANN401
        called["cls"] = cls
        called["traverser"] = traverser
        called["vars"] = kwargs.get("variables") or kwargs.get("vars") or {}

    monkeypatch.setattr(c.dropconnect_traverser, "register", fake_register, raising=True)

    importlib.reload(flax_mod)

    # registry should target our custom NNX Dense
    assert called.get("cls") is Dense

    traverser = called.get("traverser")
    assert callable(traverser)
    assert getattr(traverser, "__name__", "") == "replace_nnx_dropconnect_dense"

    vars_dict = cast(dict[str, Any], called.get("vars"))
    assert set(vars_dict.keys()) == {"p"}
    assert vars_dict["p"] is c.P


def test_if_replace_nnx_dropconnect_dense_works() -> None:
    """Replacement should preserve Dense config and set p correctly."""
    rngs = nnx.Rngs(0)

    dense = Dense(
        features=7,
        use_bias=False,
        dtype=jnp.float16,
        param_dtype=jnp.float32,
        precision=None,
        kernel_init=nnx.initializers.lecun_normal(),
        bias_init=nnx.initializers.zeros,
        rngs=rngs,
    )

    replaced = flax_mod.replace_nnx_dropconnect_dense(dense, p=0.3, rngs=nnx.Rngs(1))

    assert isinstance(replaced, DropConnectDense)
    assert replaced.features == dense.features
    assert replaced.use_bias == dense.use_bias
    assert replaced.dtype == dense.dtype
    assert replaced.precision == dense.precision
    assert replaced.kernel_init == dense.kernel_init
    assert replaced.bias_init == dense.bias_init
    assert replaced.p == 0.3
