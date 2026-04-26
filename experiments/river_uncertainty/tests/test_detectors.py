"""Tests for drift detectors."""

from __future__ import annotations

import math

from river_uq.detectors import (
    ARFNativeDetector,
    PageHinkleyErrorDetector,
    ProblyUQDetector,
)


def test_probly_uq_no_alarm_when_flat() -> None:
    det = ProblyUQDetector(warmup=(0, 100), smoothing_win=10, k=4.0, min_consec=5)
    for t in range(500):
        det.update(t, epi=0.05, error=0, model_n_drifts=0)
    assert det.first_alarm is None


def test_probly_uq_fires_on_spike() -> None:
    det = ProblyUQDetector(warmup=(0, 100), smoothing_win=10, k=2.0, min_consec=3)
    for t in range(150):
        det.update(t, epi=0.05, error=0, model_n_drifts=0)
    fired_at = None
    for t in range(150, 300):
        if det.update(t, epi=1.0, error=0, model_n_drifts=0):
            fired_at = t
            break
    assert fired_at is not None
    assert det.first_alarm == fired_at


def test_arf_native_detector_fires_when_counter_grows() -> None:
    det = ARFNativeDetector(warmup_end=10)
    for t in range(20):
        det.update(t, epi=0.0, error=0, model_n_drifts=0)
    assert det.first_alarm is None
    fired = det.update(20, epi=0.0, error=0, model_n_drifts=1)
    assert fired
    assert det.first_alarm == 20


def test_arf_native_ignores_warmup_alarms() -> None:
    det = ARFNativeDetector(warmup_end=100)
    for t in range(50):
        det.update(t, epi=0.0, error=0, model_n_drifts=t)  # spuriously growing during warmup
    assert det.first_alarm is None


def test_pagehinkley_eventually_fires_on_error_burst() -> None:
    det = PageHinkleyErrorDetector(min_instances=20, threshold=5, warmup_end=20)
    for t in range(100):
        det.update(t, epi=0.0, error=0, model_n_drifts=0)
    fired_at = None
    for t in range(100, 500):
        if det.update(t, epi=0.0, error=1, model_n_drifts=0):
            fired_at = t
            break
    assert fired_at is not None
