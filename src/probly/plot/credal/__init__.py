"""Credal set plotting (binary interval, ternary simplex, and spider/radar)."""

from probly.lazy_types import TORCH_TENSOR_LIKE

from .plot import (
    _draw_credal_set_binary,
    _draw_credal_set_spider,
    _draw_credal_set_ternary,
    plot_credal_set,
)


@_draw_credal_set_binary.delayed_register(TORCH_TENSOR_LIKE)
@_draw_credal_set_ternary.delayed_register(TORCH_TENSOR_LIKE)
@_draw_credal_set_spider.delayed_register(TORCH_TENSOR_LIKE)
def _(_: type) -> None:
    from . import _torch as _torch  # noqa: PLC0415


__all__ = ["plot_credal_set"]
