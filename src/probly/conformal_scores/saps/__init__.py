"""Conformal Prediction SAPS score implementation."""

from probly.lazy_types import JAX_VALUE, TORCH_VALUE

from ._common import SAPSScore, _saps_score_dispatch, saps_score


@_saps_score_dispatch.delayed_register(TORCH_VALUE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@_saps_score_dispatch.delayed_register(JAX_VALUE)
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415


__all__ = ["SAPSScore", "saps_score"]
