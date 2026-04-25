"""Drift detectors for the paper experiment.

All detectors expose:
- update(t, *, epi, error, model_n_drifts) -> bool (True iff this step fired)
- first_alarm: int | None (first step at which the detector fired, or None)

Detectors are stateful and one-shot — they return True only on the *first*
firing; subsequent calls return False.
"""

from __future__ import annotations

from collections import deque
from typing import Protocol

import numpy as np
from river import drift


class Detector(Protocol):
    def update(self, t: int, *, epi: float, error: int, model_n_drifts: int) -> bool: ...

    @property
    def first_alarm(self) -> int | None: ...


class ProblyUQDetector:
    """Threshold-on-rolling-mean detector reading the epistemic signal.

    Label-free: only consumes ``epi``. After a warmup window, computes
    ``mu, sigma`` of the smoothed signal and fires the first time the smoothed
    signal exceeds ``mu + k*sigma`` for ``min_consec`` consecutive steps.
    """

    def __init__(
        self,
        warmup: tuple[int, int] = (500, 1500),
        smoothing_win: int = 50,
        k: float = 3.0,
        min_consec: int = 5,
    ) -> None:
        self._warmup = warmup
        self._win = smoothing_win
        self._k = k
        self._min_consec = min_consec
        self._buf: deque[float] = deque(maxlen=smoothing_win)
        self._warmup_smoothed: list[float] = []
        self._mu: float | None = None
        self._sigma: float | None = None
        self._consec = 0
        self._first_alarm: int | None = None

    def update(self, t: int, *, epi: float, error: int, model_n_drifts: int) -> bool:
        del error, model_n_drifts
        self._buf.append(float(epi))
        smoothed = float(np.mean(self._buf))

        w_start, w_end = self._warmup
        if w_start <= t < w_end:
            self._warmup_smoothed.append(smoothed)
            return False
        if t == w_end and self._warmup_smoothed:
            arr = np.asarray(self._warmup_smoothed)
            self._mu = float(arr.mean())
            self._sigma = float(arr.std()) or 1e-6
        if t < w_end or self._mu is None or self._sigma is None:
            return False
        if self._first_alarm is not None:
            return False

        threshold = self._mu + self._k * self._sigma
        if smoothed > threshold:
            self._consec += 1
        else:
            self._consec = 0
        if self._consec >= self._min_consec:
            self._first_alarm = t
            return True
        return False

    @property
    def first_alarm(self) -> int | None:
        return self._first_alarm


class ARFNativeDetector:
    """Wraps river ARF's native per-tree ADWIN counter as a one-shot detector."""

    def __init__(self, warmup_end: int = 1500) -> None:
        self._warmup_end = warmup_end
        self._last_seen = 0
        self._first_alarm: int | None = None

    def update(self, t: int, *, epi: float, error: int, model_n_drifts: int) -> bool:
        del epi, error
        if t < self._warmup_end:
            self._last_seen = model_n_drifts
            return False
        if self._first_alarm is not None:
            return False
        if model_n_drifts > self._last_seen:
            self._last_seen = model_n_drifts
            self._first_alarm = t
            return True
        return False

    @property
    def first_alarm(self) -> int | None:
        return self._first_alarm


class PageHinkleyErrorDetector:
    """river.drift.PageHinkley fed the streaming 0/1 model error."""

    def __init__(
        self,
        min_instances: int = 500,
        threshold: float = 50.0,
        warmup_end: int = 1500,
    ) -> None:
        self._ph = drift.PageHinkley(min_instances=min_instances, threshold=threshold)
        self._warmup_end = warmup_end
        self._first_alarm: int | None = None

    def update(self, t: int, *, epi: float, error: int, model_n_drifts: int) -> bool:
        del epi, model_n_drifts
        self._ph.update(float(error))
        if t < self._warmup_end or self._first_alarm is not None:
            return False
        if self._ph.drift_detected:
            self._first_alarm = t
            return True
        return False

    @property
    def first_alarm(self) -> int | None:
        return self._first_alarm
