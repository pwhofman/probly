"""Synthetic stream builders for the paper experiment.

All streams are 3000 steps. Drift streams place an abrupt drift at t=2000.
"""

from __future__ import annotations

import itertools
import math
import random
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
    """Agrawal with abrupt **classification-function** change at ``drift_t``.

    Parameterised by ``(pre_fn, post_fn)`` and an optional ``drift_t``
    (defaults to :data:`DRIFT_T`). The input distribution is unchanged; the
    labeling rule flips at ``drift_t``. Whether this drift is detectable by
    an EU-based detector depends on the function pair.
    """

    def __init__(
        self,
        seed: int,
        pre_fn: int,
        post_fn: int,
        drift_t: int | None = None,
    ) -> None:
        self._pre = synth.Agrawal(classification_function=pre_fn, seed=seed)
        self._post = synth.Agrawal(classification_function=post_fn, seed=seed + 1)
        self._drift_t = DRIFT_T if drift_t is None else drift_t

    def take(self, n: int) -> Iterator[tuple[dict, int]]:
        position = self._drift_t
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


_AGRAWAL_DRIFT_0TO9_T: Final[int] = 1500


def _agrawal_drift_0to9(seed: int) -> _AbruptAgrawalDrift:
    """Spike-and-recover: pre-drift fn 0 is hard (epi ~0.23), post-drift fn 9
    is so simple that ARF converges to perfect agreement and epi decays to ~0
    within ~1000 steps. Uses an earlier drift point than the other variants
    (``t = _AGRAWAL_DRIFT_0TO9_T``) so that a 4 000-step run shows the full
    pre-drift baseline + post-drift recovery within one figure."""
    return _AbruptAgrawalDrift(seed, pre_fn=0, post_fn=9, drift_t=_AGRAWAL_DRIFT_0TO9_T)


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


_AGE_THRESHOLD = 40


class _AgrawalVirtualDrift:
    """Virtual (covariate) drift on Agrawal: classification function is fixed
    at fn=4 throughout, but the input distribution shifts at ``t=DRIFT_T``.

    Pre-drift: only customers with ``age >= 40`` are emitted (younger applicants
    are rejection-filtered out of the stream). Post-drift, behaviour depends
    on ``post_mode``:

    * ``"join"``: no filter -- the full age range appears, so young customers
      *join* the existing population (about a third of post-drift samples are
      now novel).
    * ``"replace"``: only ``age < 40`` is emitted -- post-drift the entire
      population is novel to the model.

    Labels (fn 4) are unchanged across the boundary. Any uncertainty rise is
    therefore purely epistemic in nature: the labelling rule is deterministic
    and identical, only the model's familiarity with the input region changes.
    """

    def __init__(self, seed: int, post_mode: str) -> None:
        if post_mode not in {"join", "replace"}:
            msg = f"post_mode must be 'join' or 'replace', got {post_mode!r}"
            raise ValueError(msg)
        self._pre = synth.Agrawal(classification_function=4, seed=seed)
        self._post = synth.Agrawal(classification_function=4, seed=seed + 1)
        self._post_mode = post_mode

    def take(self, n: int) -> Iterator[tuple[dict, int]]:
        position = DRIFT_T
        pre_iter = ((x, y) for x, y in iter(self._pre) if x["age"] >= _AGE_THRESHOLD)
        post_raw = iter(self._post)
        if self._post_mode == "join":
            post_iter: Iterator[tuple[dict, int]] = post_raw
        else:
            post_iter = ((x, y) for x, y in post_raw if x["age"] < _AGE_THRESHOLD)
        if n <= position:
            return itertools.islice(pre_iter, n)
        return itertools.chain(
            itertools.islice(pre_iter, position),
            itertools.islice(post_iter, n - position),
        )


def _agrawal_virtual_drift_join(seed: int) -> _AgrawalVirtualDrift:
    return _AgrawalVirtualDrift(seed, post_mode="join")


def _agrawal_virtual_drift_replace(seed: int) -> _AgrawalVirtualDrift:
    return _AgrawalVirtualDrift(seed, post_mode="replace")


class _AgrawalStackedVirtualDrift:
    """Stacked virtual drift: fn=4 throughout, but pre- and post-drift restrict
    to disjoint regions of (age, salary) space.

    * Pre-drift:  ``age >= 40`` AND ``salary >= 75_000`` (older, high-salary).
    * Post-drift: ``age < 40``  AND ``salary < 75_000`` (younger, low-salary).

    The post-drift region is fully novel to the model trained on the pre-drift
    region. Tests whether EU can detect virtual drift when the input shift is
    severe enough that the trees have no overlapping training coverage.
    """

    _SALARY_THRESHOLD = 75_000

    def __init__(self, seed: int) -> None:
        self._pre = synth.Agrawal(classification_function=4, seed=seed)
        self._post = synth.Agrawal(classification_function=4, seed=seed + 1)

    def take(self, n: int) -> Iterator[tuple[dict, int]]:
        position = DRIFT_T
        pre_iter = (
            (x, y)
            for x, y in iter(self._pre)
            if x["age"] >= _AGE_THRESHOLD and x["salary"] >= self._SALARY_THRESHOLD
        )
        post_iter = (
            (x, y)
            for x, y in iter(self._post)
            if x["age"] < _AGE_THRESHOLD and x["salary"] < self._SALARY_THRESHOLD
        )
        if n <= position:
            return itertools.islice(pre_iter, n)
        return itertools.chain(
            itertools.islice(pre_iter, position),
            itertools.islice(post_iter, n - position),
        )


def _agrawal_virtual_drift_stacked(seed: int) -> _AgrawalStackedVirtualDrift:
    return _AgrawalStackedVirtualDrift(seed)


class _AgrawalGradualStackedVirtualDrift:
    """Stacked virtual drift with a sigmoid-blended fade-in window.

    Same disjoint regions as :class:`_AgrawalStackedVirtualDrift`:

    * Pre region:  ``age >= 40`` AND ``salary >= 75_000``.
    * Post region: ``age < 40``  AND ``salary < 75_000``.

    Instead of switching abruptly at ``t=2000``, samples are drawn from the
    pre stream and the post stream by a Bernoulli weighted with a
    :class:`river.datasets.synth.ConceptDriftStream`-style sigmoid centred at
    ``t=1500`` with ``width=1000`` (so the visible transition spans
    ``1000..2000``). Fn=4 throughout, so any uncertainty rise is purely
    epistemic.
    """

    _SALARY_THRESHOLD = 75_000
    _POSITION = 1500
    _WIDTH = 1000

    def __init__(self, seed: int) -> None:
        self._pre = synth.Agrawal(classification_function=4, seed=seed)
        self._post = synth.Agrawal(classification_function=4, seed=seed + 1)
        self._rng = random.Random(seed)

    def take(self, n: int) -> Iterator[tuple[dict, int]]:
        pre_iter = (
            (x, y)
            for x, y in iter(self._pre)
            if x["age"] >= _AGE_THRESHOLD and x["salary"] >= self._SALARY_THRESHOLD
        )
        post_iter = (
            (x, y)
            for x, y in iter(self._post)
            if x["age"] < _AGE_THRESHOLD and x["salary"] < self._SALARY_THRESHOLD
        )
        for t in range(n):
            p_post = 1.0 / (1.0 + math.exp(-4.0 * (t - self._POSITION) / self._WIDTH))
            if self._rng.random() < p_post:
                yield next(post_iter)
            else:
                yield next(pre_iter)


def _agrawal_virtual_drift_stacked_gradual_1000(
    seed: int,
) -> _AgrawalGradualStackedVirtualDrift:
    return _AgrawalGradualStackedVirtualDrift(seed)


def _agrawal_gradual_drift_500(seed: int) -> synth.ConceptDriftStream:
    """Gradual fade from fn 0 to fn 4 over t=1500..2000 (sigmoid blend)."""
    return synth.ConceptDriftStream(
        stream=synth.Agrawal(classification_function=0, seed=seed),
        drift_stream=synth.Agrawal(classification_function=4, seed=seed + 1),
        position=1750,
        width=500,
        seed=seed,
    )


def _agrawal_gradual_drift_1000(seed: int) -> synth.ConceptDriftStream:
    """Gradual fade from fn 0 to fn 4 over t=1000..2000 (sigmoid blend)."""
    return synth.ConceptDriftStream(
        stream=synth.Agrawal(classification_function=0, seed=seed),
        drift_stream=synth.Agrawal(classification_function=4, seed=seed + 1),
        position=1500,
        width=1000,
        seed=seed,
    )


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
    "agrawal_drift_0to9": (_agrawal_drift_0to9, _AGRAWAL_DRIFT_0TO9_T),
    "agrawal_drift_4to0": (_agrawal_drift_4to0, DRIFT_T),
    "agrawal_drift_9to2": (_agrawal_drift_9to2, DRIFT_T),
    "agrawal_covariate_drift": (_agrawal_covariate_drift, DRIFT_T),
    "agrawal_gradual_drift_500": (_agrawal_gradual_drift_500, 1750),
    "agrawal_gradual_drift_1000": (_agrawal_gradual_drift_1000, 1500),
    "agrawal_virtual_drift_join": (_agrawal_virtual_drift_join, DRIFT_T),
    "agrawal_virtual_drift_replace": (_agrawal_virtual_drift_replace, DRIFT_T),
    "agrawal_virtual_drift_stacked": (_agrawal_virtual_drift_stacked, DRIFT_T),
    "agrawal_virtual_drift_stacked_gradual_1000": (
        _agrawal_virtual_drift_stacked_gradual_1000,
        1500,
    ),
    "electricity": (_electricity, None),
}

STREAM_NAMES: Final[tuple[str, ...]] = tuple(_BUILDERS)

_GRADUAL_WINDOWS: Final[dict[str, tuple[int, int]]] = {
    "agrawal_gradual_drift_500": (1500, 2000),
    "agrawal_gradual_drift_1000": (1000, 2000),
    "agrawal_virtual_drift_stacked_gradual_1000": (1000, 2000),
}


def get_drift_window(name: str) -> tuple[int, int] | None:
    """Return ``(start, end)`` of the drift fade for gradual streams, else ``None``.

    Abrupt streams (and stationary ones) return ``None`` — for them the change
    point is fully described by ``true_drift_t`` from :func:`build_stream`.
    """
    return _GRADUAL_WINDOWS.get(name)


def build_stream(
    name: str,
    seed: int,
    n: int = DEFAULT_N,
) -> tuple[Iterator[tuple[dict, int]], int | None]:
    """Build a named stream.

    Args:
        name: One of the names listed in :data:`STREAM_NAMES`.
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
