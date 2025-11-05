"""Tests for the Flax DropConnect transformation registration and replacement."""

from __future__ import annotations

import importlib
from typing import Any, cast

import pytest

jax = pytest.importorskip("jax")
from jax import numpy as jnp  # noqa: E402

from probly.layers.flax import DropConnectDense  # noqa: E402
from probly.transformation.dropconnect import common as c, flax as flax_mod  # noqa: E402

flax = pytest.importorskip("flax")
from flax import linen as nn  # noqa: E402


def test_flax_register_calls_dropconnect_traverser_on_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Import-time register should wire nn.Dense with replace_flax_dropconnect_dense and pass P."""
    called: dict[str, Any] = {}

    def fake_register(*, cls: type[Any], traverser: Any, **kwargs: Any) -> None:  # noqa: ANN401
        called["cls"] = cls
        called["traverser"] = traverser
        called["vars"] = kwargs.get("variables") or kwargs.get("vars") or {}

    # replace every call of register from the traverser with call of fake_register
    monkeypatch.setattr(c.dropconnect_traverser, "register", fake_register, raising=True)

    # reload for test if register loads correct values
    from probly.transformation.dropconnect import flax as flax_mod  # noqa: PLC0415

    importlib.reload(flax_mod)

    # check for results of register
    assert called.get("cls") is nn.Dense
    traverser = called.get("traverser")
    assert callable(traverser)
    # check name of functiom
    assert getattr(traverser, "__name__", "") == "replace_flax_dropconnect_dense"

    vars_dict = cast(dict[str, Any], called.get("vars"))
    assert set(vars_dict.keys()) == {"p"}
    assert vars_dict["p"] is c.P


def test_if_replace_flax_dropconnect_dense_works() -> None:
    """Replacement should preserve Dense config and set p correctly."""
    dense = nn.Dense(
        features=7,
        use_bias=False,
        dtype=jnp.float16,
        param_dtype=jnp.float32,
        precision=None,
        kernel_init=nn.linear.default_kernel_init,
        bias_init=nn.initializers.zeros,
    )

    replaced = flax_mod.replace_flax_dropconnect_dense(dense, p=0.3)

    assert isinstance(replaced, DropConnectDense)
    assert replaced.features == dense.features
    assert replaced.use_bias == dense.use_bias
    assert replaced.dtype == dense.dtype
    assert replaced.precision == dense.precision
    assert replaced.kernel_init == dense.kernel_init
    assert replaced.bias_init == dense.bias_init
    assert replaced.p == 0.3
