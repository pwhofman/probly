"""Wasserstein distance nonconformity score."""

from __future__ import annotations

from probly.lazy_types import JAX_VALUE, TORCH_VALUE

from ._common import WassersteinDistanceScore, wasserstein_distance_score, wasserstein_distance_score_func


@wasserstein_distance_score_func.delayed_register(TORCH_VALUE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@wasserstein_distance_score_func.delayed_register(JAX_VALUE)
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415


__all__ = ["WassersteinDistanceScore", "wasserstein_distance_score", "wasserstein_distance_score_func"]
