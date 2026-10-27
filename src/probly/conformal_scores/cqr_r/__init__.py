"""Conformal Prediction CQR-r score implementation."""

from probly.lazy_types import JAX_VALUE, TORCH_VALUE

from ._common import cqr_r_score


@cqr_r_score.delayed_register(TORCH_VALUE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@cqr_r_score.delayed_register(JAX_VALUE)
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415


__all__ = ["cqr_r_score"]
