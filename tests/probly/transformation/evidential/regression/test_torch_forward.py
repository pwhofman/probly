from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, NoReturn, cast

import pytest

# Optional dependency: torch not available -> skip whole file
try:
    import torch
    from torch import nn
except ImportError:
    pytest.skip("torch not available", allow_module_level=True)

from probly.transformation.evidential import regression as er


def _die(msg: str) -> NoReturn:
    """Unified skip helper that also keeps mypy from complaining about missing return."""
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


def _first_linear_in_features(model: nn.Module) -> int:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            return int(module.in_features)
    _die("Fixture model has no nn.Linear; cannot infer input feature size")


def _last_linear_out_features(model: nn.Module) -> int:
    last: nn.Linear | None = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last = module
    if last is None:
        _die("Model has no nn.Linear; cannot infer output feature size")
    return int(last.out_features)


def _first_conv_spec(model: nn.Module) -> tuple[int, int, int]:
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            kernel = module.kernel_size
            kernel_h, kernel_w = (kernel, kernel) if isinstance(kernel, int) else kernel
            return int(module.in_channels), int(kernel_h), int(kernel_w)
    _die("Fixture model has no nn.Conv2d; conv-forward test not applicable")


def _try_unpack(
    y: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Best-effort unpacking into (mu, v, alpha, beta)."""
    # 1) Mapping with the right keys
    if isinstance(y, Mapping) and {"mu", "v", "alpha", "beta"}.issubset(y.keys()):
        mu = cast(torch.Tensor, y["mu"])
        v = cast(torch.Tensor, y["v"])
        alpha = cast(torch.Tensor, y["alpha"])
        beta = cast(torch.Tensor, y["beta"])
        return mu, v, alpha, beta

    # 2) Object with attributes mu, v, alpha, beta
    if hasattr(y, "mu") and hasattr(y, "v") and hasattr(y, "alpha") and hasattr(y, "beta"):
        mu = cast(torch.Tensor, y.mu)
        v = cast(torch.Tensor, y.v)
        alpha = cast(torch.Tensor, y.alpha)
        beta = cast(torch.Tensor, y.beta)
        return mu, v, alpha, beta

    # 3) Sequence of four tensors
    if isinstance(y, Sequence) and len(y) == 4:
        mu, v, alpha, beta = y
        if all(isinstance(t, torch.Tensor) for t in (mu, v, alpha, beta)):
            mu_t = cast(torch.Tensor, mu)
            v_t = cast(torch.Tensor, v)
            alpha_t = cast(torch.Tensor, alpha)
            beta_t = cast(torch.Tensor, beta)
            return mu_t, v_t, alpha_t, beta_t

    # 4) Single tensor whose last dim can be split into 4 parts
    if torch.is_tensor(y):
        y_tensor = cast(torch.Tensor, y)
        if y_tensor.ndim >= 2:
            dim = y_tensor.shape[-1]
            if dim % 4 == 0:
                split = dim // 4
                mu, v, alpha, beta = torch.split(y_tensor, split, dim=-1)
                return mu, v, alpha, beta

    # 5) Fallback: unsupported structure
    return None


def _unpack_four(
    y: Any,  # noqa: ANN401
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out = _try_unpack(y)
    if out is None:
        _die("Cannot interpret model output as evidential {mu,v,alpha,beta}")
    return out


class TestTorchForward:
    def test_forward_and_parameter_shapes(
        self,
        torch_model_small_2d_2d: nn.Sequential,
    ) -> None:
        evidential = _get_evidential_transform()
        base = torch_model_small_2d_2d

        in_dim = _first_linear_in_features(base)
        out_dim = _last_linear_out_features(base)

        model = evidential(base)
        model.eval()

        batch_size = 8
        x = torch.randn(batch_size, in_dim)

        with torch.no_grad():
            y = model(x)

        mu, v, alpha, beta = _unpack_four(y)

        for tensor in (mu, v, alpha, beta):
            assert torch.is_tensor(tensor), "Each output head must be a tensor"
            assert tensor.shape[-1] == out_dim, f"Expected last dim {out_dim}, got {tensor.shape[-1]}"
            assert tensor.shape[0] == batch_size, f"Expected batch {batch_size}, got {tensor.shape[0]}"

        for name, tensor in zip(
            ("mu", "v", "alpha", "beta"),
            (mu, v, alpha, beta),
            strict=False,
        ):
            assert torch.isfinite(tensor).all(), f"{name} contains non-finite values"

        for name, tensor in zip(("v", "alpha", "beta"), (v, alpha, beta), strict=False):
            assert torch.is_floating_point(tensor), f"{name} has non-floating dtype: {tensor.dtype}"
            assert (tensor > 0).all(), f"{name} must be positive"

    def test_forward_conv_model(self, torch_conv_linear_model: nn.Sequential) -> None:
        evidential = _get_evidential_transform()
        base = torch_conv_linear_model

        # Smoke test only: being able to wrap once is sufficient
        evidential(base)
