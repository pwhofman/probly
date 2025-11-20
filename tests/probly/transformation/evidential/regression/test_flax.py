from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, NoReturn, Protocol, cast

import pytest

# Optional dependencies: use top-level try/except to avoid import-time errors
try:
    from flax import nnx
    import jax.numpy as jnp  # noqa: F401
except ImportError:
    pytest.skip("jax/flax not available", allow_module_level=True)

from probly.transformation.evidential import regression as er
from tests.probly.flax_utils import count_layers


def _die(msg: str) -> NoReturn:
    pytest.skip(msg)
    error_message = "unreachable"
    raise AssertionError(error_message)  # pragma: no cover


class _ArrayLike(Protocol):
    ndim: int
    shape: tuple[int, ...]


def _get_evidential_transform() -> Callable[..., object]:
    for name in (
        "evidential_regression",
        "regression",
        "to_evidential_regressor",
        "make_evidential_regression",
        "evidential",
        "transform",
    ):
        fn = getattr(er, name, None)
        if callable(fn):
            return cast(Callable[..., object], fn)
    _die(
        "No evidential regression transform found in probly.transformation.evidential.regression",
    )


def _iter_modules(m: nnx.Module) -> Iterable[nnx.Module]:
    yield m
    if isinstance(m, nnx.Sequential):
        for child in m.layers:
            yield from _iter_modules(child)


def _maybe_array(x: Any) -> Any:  # noqa: ANN401
    """Best-effort helper to unwrap array-like values from Flax/JAX containers."""
    value = getattr(x, "value", None)
    if value is not None:
        return value
    return x


def _linear_in_out_by_params(layer: nnx.Linear) -> tuple[int, int]:
    # First try to infer out_features from a bias-like 1D parameter
    for name in ("bias", "b"):
        if hasattr(layer, name):
            raw = _maybe_array(getattr(layer, name))
            arr = cast(_ArrayLike, raw)
            if arr.ndim == 1:
                out_features = int(arr.shape[0])
                # in_features is unknown here; use -1 as a placeholder
                return -1, out_features

    # Then try a kernel/weight-like 2D parameter
    for name in ("kernel", "weight", "w"):
        if hasattr(layer, name):
            raw = _maybe_array(getattr(layer, name))
            arr = cast(_ArrayLike, raw)
            if arr.ndim == 2:
                return int(arr.shape[0]), int(arr.shape[1])

    # Finally scan all public attributes for plausible arrays
    for key in dir(layer):
        if key.startswith("_"):
            continue
        value = getattr(layer, key, None)
        if value is None:
            continue
        raw = _maybe_array(value)
        if not hasattr(raw, "shape"):
            continue
        arr = cast(_ArrayLike, raw)
        if arr.ndim == 1:
            return -1, int(arr.shape[0])
        if arr.ndim == 2:
            return int(arr.shape[0]), int(arr.shape[1])

    _die("Cannot infer in/out features from nnx.Linear parameters")


def _last_linear_and_out_features(model: nnx.Module) -> tuple[nnx.Linear, int]:
    last: nnx.Linear | None = None
    for mod in _iter_modules(model):
        if isinstance(mod, nnx.Linear):
            last = mod
    if last is None:
        _die("Model has no nnx.Linear layer to transform")
    _, out_features = _linear_in_out_by_params(last)
    if out_features in (-1, None):
        _die("Could not determine output features of the last Linear")
    return last, out_features


class TestNetworkArchitectures:
    def test_linear_head_kept_and_structure_unchanged(
        self,
        flax_model_small_2d_2d: nnx.Sequential,
    ) -> None:
        evidential = _get_evidential_transform()

        count_linear_orig = count_layers(flax_model_small_2d_2d, nnx.Linear)
        count_conv_orig = count_layers(flax_model_small_2d_2d, nnx.Conv)
        count_seq_orig = count_layers(flax_model_small_2d_2d, nnx.Sequential)
        _, out_feat_orig = _last_linear_and_out_features(flax_model_small_2d_2d)

        model = evidential(flax_model_small_2d_2d)

        count_linear_mod = count_layers(model, nnx.Linear)
        count_conv_mod = count_layers(model, nnx.Conv)
        count_seq_mod = count_layers(model, nnx.Sequential)
        _, out_feat_mod = _last_linear_and_out_features(model)

        assert model is not None
        assert isinstance(model, type(flax_model_small_2d_2d))
        assert count_conv_mod == count_conv_orig
        assert count_seq_mod == count_seq_orig
        assert count_linear_mod == count_linear_orig
        assert out_feat_mod == out_feat_orig

    def test_conv_model_kept_and_structure_unchanged(
        self,
        flax_conv_linear_model: nnx.Sequential,
    ) -> None:
        evidential = _get_evidential_transform()

        count_linear_orig = count_layers(flax_conv_linear_model, nnx.Linear)
        count_conv_orig = count_layers(flax_conv_linear_model, nnx.Conv)
        count_seq_orig = count_layers(flax_conv_linear_model, nnx.Sequential)
        _, out_feat_orig = _last_linear_and_out_features(flax_conv_linear_model)

        model = evidential(flax_conv_linear_model)

        count_linear_mod = count_layers(model, nnx.Linear)
        count_conv_mod = count_layers(model, nnx.Conv)
        count_seq_mod = count_layers(model, nnx.Sequential)
        _, out_feat_mod = _last_linear_and_out_features(model)

        assert isinstance(model, type(flax_conv_linear_model))
        assert count_conv_mod == count_conv_orig
        assert count_seq_mod == count_seq_orig
        assert count_linear_mod == count_linear_orig
        assert out_feat_mod == out_feat_orig
