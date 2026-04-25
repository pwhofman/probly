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


class _AbruptAgrawalDrift:
    """Agrawal with abrupt **classification-function** change at t=DRIFT_T.

    Parameterised by ``(pre_fn, post_fn)``. The input distribution is
    unchanged; the labeling rule flips at ``DRIFT_T``. Whether this drift is
    detectable by an EU-based detector depends on the function pair (see
    ``scripts/scan_agrawal_functions.py`` for an empirical map).
    """

    def __init__(self, seed: int, pre_fn: int, post_fn: int) -> None:
        self._pre = synth.Agrawal(classification_function=pre_fn, seed=seed)
        self._post = synth.Agrawal(classification_function=post_fn, seed=seed + 1)

    def take(self, n: int) -> Iterator[tuple[dict, int]]:
        position = DRIFT_T
        if n <= position:
            return itertools.islice(iter(self._pre), n)
        return itertools.chain(
            itertools.islice(iter(self._pre), position),
            itertools.islice(iter(self._post), n - position),
        )


def _agrawal_drift(seed: int) -> _AbruptAgrawalDrift:
    # default: 0 -> 4 (label-flip; EU does NOT fire on deep methods)
    return _AbruptAgrawalDrift(seed, pre_fn=0, post_fn=4)


def _agrawal_drift_7to4(seed: int) -> _AbruptAgrawalDrift:
    """EU works: members disagree on post-drift inputs (epi 0.05 -> 0.28)."""
    return _AbruptAgrawalDrift(seed, pre_fn=7, post_fn=4)


def _agrawal_drift_4to0(seed: int) -> _AbruptAgrawalDrift:
    """EU fails: members confidently track wrong answer together
    (acc drops ~0.44 yet epi drops too -- the 'confidently wrong' regime)."""
    return _AbruptAgrawalDrift(seed, pre_fn=4, post_fn=0)


def _agrawal_drift_9to2(seed: int) -> _AbruptAgrawalDrift:
    """EU works dramatically: pre-drift function 9 is so simple deep
    ensembles converge to perfect agreement (epi=0); any change explodes
    disagreement (epi 0.0 -> 0.30)."""
    return _AbruptAgrawalDrift(seed, pre_fn=9, post_fn=2)


# Agrawal feature ranges (per river docs / source):
#   salary ~ U(20000, 150000)   age ~ U(20, 80)
# We shift salary up by +80000 post-drift so that post-drift inputs land
# in a region the pre-drift model never trained on. This is genuine
# covariate shift while keeping the same labeling rule.
_AGRAWAL_SALARY_SHIFT = 80000.0


class _AbruptAgrawalCovariateDrift:
    """Agrawal with abrupt **covariate** drift at t=DRIFT_T.

    Same ``classification_function=0`` throughout, but post-drift the
    ``salary`` feature is shifted by ``_AGRAWAL_SALARY_SHIFT``. The labeling
    rule is unchanged; the input distribution shifts. Designed to be
    detectable by model-disagreement-based UQ on neural networks.
    """

    def __init__(self, seed: int) -> None:
        self._pre = synth.Agrawal(classification_function=0, seed=seed)
        self._post = synth.Agrawal(classification_function=0, seed=seed + 1)

    @staticmethod
    def _shift(sample: tuple[dict, int]) -> tuple[dict, int]:
        x, y = sample
        x_shift = dict(x)
        x_shift["salary"] = float(x_shift["salary"]) + _AGRAWAL_SALARY_SHIFT
        return x_shift, y

    def take(self, n: int) -> Iterator[tuple[dict, int]]:
        position = DRIFT_T
        if n <= position:
            return itertools.islice(iter(self._pre), n)
        return itertools.chain(
            itertools.islice(iter(self._pre), position),
            (self._shift(s) for s in itertools.islice(iter(self._post), n - position)),
        )


def _agrawal_covariate_drift(seed: int) -> _AbruptAgrawalCovariateDrift:
    return _AbruptAgrawalCovariateDrift(seed)


class _ElectricityWrapper:
    """Real-world Elec2 dataset (no labeled drift point).

    Casts the boolean label to int and exposes a uniform ``.take(n)`` API.
    """

    def __init__(self, seed: int) -> None:  # noqa: ARG002 - real data, seed unused
        from river.datasets import Elec2

        self._stream = Elec2()

    def take(self, n: int) -> Iterator[tuple[dict, int]]:
        return ((x, int(y)) for x, y in itertools.islice(iter(self._stream), n))


def _electricity(seed: int) -> _ElectricityWrapper:
    return _ElectricityWrapper(seed)


_BUILDERS = {
    "stagger_drift": (_stagger_drift, DRIFT_T),
    "sea_drift": (_sea_drift, DRIFT_T),
    "agrawal_stationary": (_agrawal_stationary, None),
    "agrawal_drift": (_agrawal_drift, DRIFT_T),
    "agrawal_drift_7to4": (_agrawal_drift_7to4, DRIFT_T),
    "agrawal_drift_4to0": (_agrawal_drift_4to0, DRIFT_T),
    "agrawal_drift_9to2": (_agrawal_drift_9to2, DRIFT_T),
    "agrawal_covariate_drift": (_agrawal_covariate_drift, DRIFT_T),
    "electricity": (_electricity, None),
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
