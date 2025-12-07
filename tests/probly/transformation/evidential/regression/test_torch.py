from __future__ import annotations

from collections.abc import Callable
from typing import Any, NoReturn, cast

import pytest

# Optional dependency
try:
    import torch  # noqa: F401
    from torch import nn
except ImportError:
    pytest.skip("torch not available", allow_module_level=True)

from probly.transformation.evidential import regression as er
from tests.probly.torch_utils import count_layers


def _die(msg: str) -> NoReturn:
    pytest.skip(msg)
    error_message = "unreachable"
    raise AssertionError(error_message)  # pragma: no cover


def _get_evidential_transform() -> Callable[..., Any]:
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
            return cast(Callable[..., Any], fn)
    _die(
        "No evidential regression transform found in probly.transformation.evidential.regression",
    )


def _last_linear_and_out_features(model: nn.Module) -> tuple[nn.Linear, int]:
    last: nn.Linear | None = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last = module
    if last is None:
        _die("Model has no nn.Linear layer to transform")
    return last, int(last.out_features)


def _last_module(model: nn.Module) -> nn.Module:
    last: nn.Module | None = None
    for module in model.modules():
        last = module
    assert last is not None  # for mypy
    return last


class TestNetworkArchitectures:
    def test_linear_head_kept_or_replaced_once_and_structure_ok(
        self,
        torch_model_small_2d_2d: nn.Sequential,
    ) -> None:
        evidential = _get_evidential_transform()

        count_linear_orig = count_layers(torch_model_small_2d_2d, nn.Linear)
        count_conv_orig = count_layers(torch_model_small_2d_2d, nn.Conv2d)
        count_seq_orig = count_layers(torch_model_small_2d_2d, nn.Sequential)
        _, out_feat_orig = _last_linear_and_out_features(torch_model_small_2d_2d)

        model = evidential(torch_model_small_2d_2d)

        count_linear_mod = count_layers(model, nn.Linear)
        count_conv_mod = count_layers(model, nn.Conv2d)
        count_seq_mod = count_layers(model, nn.Sequential)
        _, out_feat_mod = _last_linear_and_out_features(model)

        assert model is not None
        assert isinstance(model, type(torch_model_small_2d_2d))
        assert count_conv_mod == count_conv_orig
        assert count_seq_mod == count_seq_orig

        assert count_linear_mod <= count_linear_orig
        assert (count_linear_orig - count_linear_mod) in (0, 1)

        if count_linear_mod == count_linear_orig - 1:
            tail = _last_module(model)
            assert not isinstance(tail, nn.Linear)

        assert out_feat_mod == out_feat_orig

    def test_conv_model_kept_or_replaced_once_and_structure_ok(
        self,
        torch_conv_linear_model: nn.Sequential,
    ) -> None:
        evidential = _get_evidential_transform()

        count_linear_orig = count_layers(torch_conv_linear_model, nn.Linear)
        count_conv_orig = count_layers(torch_conv_linear_model, nn.Conv2d)
        count_seq_orig = count_layers(torch_conv_linear_model, nn.Sequential)
        _, out_feat_orig = _last_linear_and_out_features(torch_conv_linear_model)

        model = evidential(torch_conv_linear_model)

        count_linear_mod = count_layers(model, nn.Linear)
        count_conv_mod = count_layers(model, nn.Conv2d)
        count_seq_mod = count_layers(model, nn.Sequential)
        _, out_feat_mod = _last_linear_and_out_features(model)

        assert isinstance(model, type(torch_conv_linear_model))
        assert count_conv_mod == count_conv_orig
        assert count_seq_mod == count_seq_orig

        assert count_linear_mod <= count_linear_orig
        assert (count_linear_orig - count_linear_mod) in (0, 1)

        if count_linear_mod == count_linear_orig - 1:
            tail = _last_module(model)
            assert not isinstance(tail, nn.Linear)

        assert out_feat_mod == out_feat_orig
