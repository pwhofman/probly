"""Synthetic data streams for the ARF uncertainty experiments.

We primarily use river's built-in synthetic generators because they already
return dictionaries of features, which plays nicely with river's online
estimators. The :func:`make_synthetic_stream` helper centralises the stream
choice so experiments can switch between stationary, abrupt-drift and
gradual-drift settings without touching their main loop.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

from river.datasets import base, synth

type StreamKind = Literal["agrawal", "sea_drift", "rbf_drift", "stagger_drift"]


def make_synthetic_stream(
    kind: StreamKind = "sea_drift",
    n_samples: int = 5_000,
    seed: int = 0,
) -> Iterator[tuple[dict[str, float], int]]:
    """Return an iterator over ``(features, label)`` tuples.

    Args:
        kind: Which synthetic stream to use. ``"agrawal"`` is stationary,
            the ``*_drift`` options compose two base streams via
            :class:`river.datasets.synth.ConceptDriftStream` with a sigmoidal
            transition centred around the middle of the stream.
        n_samples: How many samples to draw from the stream.
        seed: Random seed used by every underlying generator.

    Yields:
        Pairs of ``(features, label)`` exactly as river produces them.
    """
    stream = _build_stream(kind, n_samples=n_samples, seed=seed)
    yield from stream.take(n_samples)


def _build_stream(kind: StreamKind, *, n_samples: int, seed: int) -> base.Dataset:
    if kind == "agrawal":
        return synth.Agrawal(seed=seed)

    if kind == "sea_drift":
        # SEA uses the sum of two features + threshold; variant 0 has a small
        # threshold (8) and variant 3 the largest (9.5) which yields the most
        # dramatic shift of the decision boundary.
        return synth.ConceptDriftStream(
            stream=synth.SEA(variant=0, seed=seed),
            drift_stream=synth.SEA(variant=3, seed=seed + 1),
            position=n_samples // 2,
            width=30,
            seed=seed,
        )

    if kind == "rbf_drift":
        return synth.RandomRBFDrift(
            seed_model=seed,
            seed_sample=seed + 1,
            n_classes=3,
            n_features=5,
            n_centroids=10,
            change_speed=0.001,
            n_drift_centroids=4,
        )

    if kind == "stagger_drift":
        return synth.ConceptDriftStream(
            stream=synth.STAGGER(classification_function=0, seed=seed),
            drift_stream=synth.STAGGER(classification_function=2, seed=seed + 1),
            position=n_samples // 2,
            width=max(1, n_samples // 100),
            seed=seed,
        )

    msg = f"Unknown stream kind: {kind!r}"
    raise ValueError(msg)
