"""Numpy array function implementations for sample arrays."""

from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING, Any

import numpy as np

from probly.utils import switchdispatch

if TYPE_CHECKING:
    from collections.abc import Callable


@switchdispatch
def array_function(
    func: Callable,
    types: tuple[type[Any], ...],  # noqa: ARG001
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Implementation of numpy array functions for sample arrays."""
    return func._implementation(*args, **kwargs)  # type: ignore[attr-defined]  # noqa: SLF001


@array_function.multi_register(
    [
        np.argmax,
        np.argmin,
        np.average,
        np.count_nonzero,
        np.mean,
        np.median,
        np.nanargmax,
        np.nanargmin,
        np.nanmax,
        np.nanmedian,
        np.nanmin,
        np.nanprod,
        np.nanstd,
        np.nansum,
        np.nanvar,
        np.std,
        np.var,
    ],
)
def array_reduction_function(
    func: Callable,
    types: tuple[type[Any], ...],  # noqa: ARG001
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401
    """Implementation of dimension-reducing array functions.

    Functions
    """
    sig = signature(func)
    params = sig.bind(*args, **kwargs)
    params.apply_defaults()

    out = params.arguments.get("out", None)
    axis = params.arguments.get("axis", None)
    keepdims = params.arguments.get("keepdims", False)


# [
#     np.argsort,
#     np.cumprod,
#     np.cumsum,
#     np.cumulative_prod,
#     np.cumulative_sum,
#     np.diff,
#     np.gradient,
#     np.nancumprod,
#     np.nancumsum,
#     np.packbits,
#     np.sort,
#     np.trapezoid,
#     np.unwrap,
# ]
