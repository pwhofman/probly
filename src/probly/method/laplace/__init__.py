"""Laplace approximation method."""

from __future__ import annotations

from probly.lazy_types import LAPLACE_BASE
from probly.representer._representer import representer


@representer.delayed_register(LAPLACE_BASE)
def _(_: type) -> None:
    from . import torch as torch  # noqa: PLC0415


__all__ = ["LaplaceRepresenter"]
