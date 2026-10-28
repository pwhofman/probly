"""Conformal Prediction absolute error score implementation."""

from probly.lazy_types import JAX_VALUE, TORCH_VALUE

from ._common import absolute_error_score


@absolute_error_score.delayed_register(TORCH_VALUE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@absolute_error_score.delayed_register(JAX_VALUE)
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415


__all__ = ["absolute_error_score"]
