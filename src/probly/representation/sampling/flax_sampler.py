"""Sampling preparation for flax."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx

from . import sampler

if TYPE_CHECKING:
    from lazy_dispatch.isinstance import LazyType
    from pytraverse import State


def _enforce_train_mode(obj: nnx.Module, state: State) -> tuple[nnx.Module, State]:
    if getattr(obj, "deterministic", False):
        obj.deterministic = False  # type: ignore[attr-defined]
        state[sampler.CLEANUP_FUNCS].add(lambda: setattr(obj, "deterministic", True))
        return obj, state
    return obj, state


def register_forced_train_mode(cls: LazyType) -> None:
    """Register a class to be forced into train mode during sampling.

    This enables Monte Carlo sampling techniques like MC Dropout :cite:`galDropoutBayesian2016`.
    """
    sampler.sampling_preparation_traverser.register(
        cls,
        _enforce_train_mode,
    )


register_forced_train_mode(
    nnx.Dropout,
)
