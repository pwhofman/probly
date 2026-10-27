"""Conformal Prediction LAC score implementation."""

from probly.lazy_types import JAX_VALUE, TORCH_VALUE

from ._common import lac_score


@lac_score.delayed_register(TORCH_VALUE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@lac_score.delayed_register(JAX_VALUE)
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415


__all__ = ["lac_score"]
