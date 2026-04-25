"""Synthetic stream builders for the paper experiment.

All streams are 3000 steps. Drift streams place an abrupt drift at t=2000.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Final

from river.datasets import synth

DRIFT_T: Final[int] = 2000
DEFAULT_N: Final[int] = 3000


class _AbruptStaggerDrift:
    """STAGGER stream with a true step change at ``position``.

    Yields ``position`` samples from the pre-drift STAGGER, then the rest from
    the post-drift STAGGER. Avoids ``ConceptDriftStream``'s sigmoid blend, which
    overflows at small ``width``.
    """

    def __init__(self, seed: int) -> None:
        self._pre = synth.STAGGER(classification_function=0, seed=seed, balance_classes=False)
        self._post = synth.STAGGER(classification_function=2, seed=seed + 1, balance_classes=False)

    def take(self, n: int) -> Iterator[tuple[dict, int]]:
        position = DRIFT_T
        if n <= position:
            return itertools.islice(iter(self._pre), n)
        return itertools.chain(
            itertools.islice(iter(self._pre), position),
            itertools.islice(iter(self._post), n - position),
        )


def _stagger_drift(seed: int) -> _AbruptStaggerDrift:
    return _AbruptStaggerDrift(seed)


def _sea_drift(seed: int) -> synth.ConceptDriftStream:
    return synth.ConceptDriftStream(
        stream=synth.SEA(variant=0, seed=seed, noise=0.1),
        drift_stream=synth.SEA(variant=3, seed=seed + 1, noise=0.1),
        position=DRIFT_T,
        width=30,
        seed=seed,
    )


def _agrawal_stationary(seed: int) -> synth.Agrawal:
    return synth.Agrawal(classification_function=0, seed=seed)


_BUILDERS = {
    "stagger_drift": (_stagger_drift, DRIFT_T),
    "sea_drift": (_sea_drift, DRIFT_T),
    "agrawal_stationary": (_agrawal_stationary, None),
}


def build_stream(
    name: str,
    seed: int,
    n: int = DEFAULT_N,
) -> tuple[Iterator[tuple[dict, int]], int | None]:
    """Build a named stream.

    Args:
        name: One of ``"stagger_drift"``, ``"sea_drift"``, ``"agrawal_stationary"``.
        seed: Stream seed (also used to seed the post-drift sub-stream as ``seed+1``).
        n: Number of samples to take.

    Returns:
        A tuple ``(iterator, true_drift_t)``. ``true_drift_t`` is ``None`` for
        stationary streams.
    """
    if name not in _BUILDERS:
        msg = f"unknown stream: {name!r} (known: {sorted(_BUILDERS)})"
        raise ValueError(msg)
    builder, drift_t = _BUILDERS[name]
    return iter(builder(seed).take(n)), drift_t
