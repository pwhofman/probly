"""Conformal set representations."""

from __future__ import annotations

from probly.lazy_types import JAX_VALUE, TORCH_VALUE

from ._common import (
    ConformalSet,
    IntervalConformalSet,
    OneHotConformalSet,
    create_interval_conformal_set,
    create_onehot_conformal_set,
)
from .numpy import NumpyIntervalConformalSet, NumpyOneHotConformalSet


@create_onehot_conformal_set.delayed_register(TORCH_VALUE)
@create_interval_conformal_set.delayed_register(TORCH_VALUE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@create_onehot_conformal_set.delayed_register(JAX_VALUE)
@create_interval_conformal_set.delayed_register(JAX_VALUE)
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415


__all__ = [
    "ConformalSet",
    "IntervalConformalSet",
    "NumpyIntervalConformalSet",
    "NumpyOneHotConformalSet",
    "OneHotConformalSet",
    "create_interval_conformal_set",
    "create_onehot_conformal_set",
]
