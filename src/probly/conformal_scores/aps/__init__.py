"""Conformal Prediction APS score implementation."""

from probly.lazy_types import JAX_VALUE, TORCH_VALUE

from ._common import APSScore, _aps_score_dispatch, aps_score


# Lazy registration - these will only be imported when needed
@_aps_score_dispatch.delayed_register(TORCH_VALUE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@_aps_score_dispatch.delayed_register(JAX_VALUE)
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415


__all__ = ["APSScore", "aps_score"]
