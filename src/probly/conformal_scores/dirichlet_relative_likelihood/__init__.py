"""Dirichlet relative likelihood non-conformity score."""

from __future__ import annotations

from probly.lazy_types import JAX_ARRAY, JAX_ARRAY_LIKE, TORCH_TENSOR, TORCH_TENSOR_LIKE

from ._common import DirichletRLScore, dirichlet_rl_score, dirichlet_rl_score_func


@dirichlet_rl_score_func.delayed_register((TORCH_TENSOR, TORCH_TENSOR_LIKE))
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


@dirichlet_rl_score_func.delayed_register((JAX_ARRAY, JAX_ARRAY_LIKE))
def _(_: type) -> None:
    from . import jax as jax  # noqa: PLC0415


__all__ = ["DirichletRLScore", "dirichlet_rl_score", "dirichlet_rl_score_func"]
