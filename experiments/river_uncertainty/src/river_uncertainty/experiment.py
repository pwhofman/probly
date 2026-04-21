"""Prequential experiment loop for ARF + probly uncertainty quantification.

The experiment follows the classic online-learning protocol:

    for each (x, y) in the stream:
        1. snapshot the ARF's per-tree distributions on x
        2. compute uncertainty scores from that snapshot
        3. record prediction correctness (pre-update, aka prequential)
        4. call ``arf.learn_one(x, y)``

This way the uncertainty at step ``t`` genuinely reflects what the model knew
*before* seeing ``(x_t, y_t)``, which is what we want when studying drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from probly.quantification import SecondOrderEntropyDecomposition, SecondOrderZeroOneDecomposition
from river_uncertainty.plotting import rolling_mean
from river_uncertainty.representation import ARFEnsembleRepresentation, river_arf_to_probly_sample

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from river import forest

    from probly.representation.distribution.array_categorical import (
        ArrayCategoricalDistributionSample,
    )


@dataclass(slots=True)
class PrequentialTrace:
    """Per-step records accumulated while running the experiment."""

    step: list[int] = field(default_factory=list)
    y_true: list[object] = field(default_factory=list)
    y_pred: list[object] = field(default_factory=list)
    correct: list[bool] = field(default_factory=list)
    total_entropy: list[float] = field(default_factory=list)
    aleatoric_entropy: list[float] = field(default_factory=list)
    epistemic_entropy: list[float] = field(default_factory=list)
    total_zero_one: list[float] = field(default_factory=list)
    aleatoric_zero_one: list[float] = field(default_factory=list)
    epistemic_zero_one: list[float] = field(default_factory=list)
    bma_max_prob: list[float] = field(default_factory=list)
    n_drifts_detected: list[int] = field(default_factory=list)

    def as_arrays(self) -> dict[str, np.ndarray]:
        """Stack the per-step lists into numpy arrays for plotting/analysis."""
        return {
            "step": np.asarray(self.step),
            "correct": np.asarray(self.correct, dtype=bool),
            "total_entropy": np.asarray(self.total_entropy),
            "aleatoric_entropy": np.asarray(self.aleatoric_entropy),
            "epistemic_entropy": np.asarray(self.epistemic_entropy),
            "total_zero_one": np.asarray(self.total_zero_one),
            "aleatoric_zero_one": np.asarray(self.aleatoric_zero_one),
            "epistemic_zero_one": np.asarray(self.epistemic_zero_one),
            "bma_max_prob": np.asarray(self.bma_max_prob),
            "n_drifts_detected": np.asarray(self.n_drifts_detected),
        }


def run_prequential(
    arf: forest.ARFClassifier,
    stream: Iterable[tuple[dict[str, float], object]],
    *,
    warmup: int = 50,
    record_every: int = 1,
) -> PrequentialTrace:
    """Train ``arf`` online on ``stream`` and log uncertainty scores.

    Args:
        arf: A fresh river ARF classifier; will be mutated in-place.
        stream: Iterable of ``(features, label)`` tuples.
        warmup: Skip uncertainty logging for the first ``warmup`` samples; the
            ensemble is mostly empty there and the numbers are dominated by
            uniform fallbacks.
        record_every: Log every ``k``-th sample. Training still happens on
            every sample.

    Returns:
        A :class:`PrequentialTrace` with per-step metrics.
    """
    trace = PrequentialTrace()

    for step, (x, y) in enumerate(stream):
        if step >= warmup and step % record_every == 0:
            rep = river_arf_to_probly_sample(arf, x)
            _record(trace, step, y=y, arf=arf, rep=rep)

        arf.learn_one(x, y)

    return trace


def _append_to_trace(
    trace: PrequentialTrace,
    step: int,
    *,
    y: object,
    sample: ArrayCategoricalDistributionSample,
    classes: tuple[object, ...],
    bma_probs: np.ndarray,
    n_drifts: int,
) -> None:
    """Shared helper: decompose *sample* and append one row to *trace*."""
    entropy_decomp = SecondOrderEntropyDecomposition(sample)
    zero_one_decomp = SecondOrderZeroOneDecomposition(sample)

    y_pred_idx = int(np.argmax(bma_probs))
    y_pred = classes[y_pred_idx]

    trace.step.append(step)
    trace.y_true.append(y)
    trace.y_pred.append(y_pred)
    trace.correct.append(y_pred == y)
    trace.total_entropy.append(float(entropy_decomp.total))
    trace.aleatoric_entropy.append(float(entropy_decomp.aleatoric))
    trace.epistemic_entropy.append(float(entropy_decomp.epistemic))
    trace.total_zero_one.append(float(zero_one_decomp.total))
    trace.aleatoric_zero_one.append(float(zero_one_decomp.aleatoric))
    trace.epistemic_zero_one.append(float(zero_one_decomp.epistemic))
    trace.bma_max_prob.append(float(bma_probs[y_pred_idx]))
    trace.n_drifts_detected.append(n_drifts)


def _record(
    trace: PrequentialTrace,
    step: int,
    *,
    y: object,
    arf: forest.ARFClassifier,
    rep: ARFEnsembleRepresentation,
) -> None:
    bma = rep.bma().probabilities
    _append_to_trace(
        trace, step,
        y=y,
        sample=rep.sample,
        classes=rep.classes,
        bma_probs=bma,
        n_drifts=int(arf.n_drifts_detected()),
    )


# ---------------------------------------------------------------------------
# Generic (model-agnostic) prequential loop
# ---------------------------------------------------------------------------


def run_prequential_generic(
    learn_fn: Callable[[dict[str, float], object], None],
    predict_fn: Callable[[dict[str, float]], tuple[ArrayCategoricalDistributionSample, tuple[object, ...]]],
    stream: Iterable[tuple[dict[str, float], object]],
    *,
    warmup: int = 50,
    record_every: int = 1,
) -> PrequentialTrace:
    """Model-agnostic prequential loop.

    Args:
        learn_fn: Called as ``learn_fn(x, y)`` to update the model(s).
        predict_fn: Called as ``predict_fn(x)`` and must return
            ``(sample, classes)`` where *sample* is an
            ``ArrayCategoricalDistributionSample`` and *classes* is the
            ordered class tuple indexing its last axis.
        stream: Iterable of ``(features, label)`` tuples.
        warmup: Skip logging for the first *warmup* samples.
        record_every: Log every k-th sample.

    Returns:
        A :class:`PrequentialTrace` with per-step metrics.
    """
    trace = PrequentialTrace()

    for step, (x, y) in enumerate(stream):
        if step >= warmup and step % record_every == 0:
            sample, classes = predict_fn(x)
            _record_generic(trace, step, y=y, sample=sample, classes=classes)

        learn_fn(x, y)

    return trace


def _record_generic(
    trace: PrequentialTrace,
    step: int,
    *,
    y: object,
    sample: ArrayCategoricalDistributionSample,
    classes: tuple[object, ...],
) -> None:
    bma = sample.array.probabilities.mean(axis=0)
    total = bma.sum()
    if total > 0:
        bma = bma / total
    _append_to_trace(
        trace, step,
        y=y,
        sample=sample,
        classes=classes,
        bma_probs=bma,
        n_drifts=0,
    )


# ---------------------------------------------------------------------------
# Drift detection helpers
# ---------------------------------------------------------------------------


def detect_drift(
    epistemic: np.ndarray,
    steps: np.ndarray,
    *,
    rolling_window: int = 30,
    baseline_window: tuple[int, int] = (500, 1_500),
    k_sigma: float = 4.0,
    min_consecutive: int = 5,
) -> tuple[int | None, float, float, np.ndarray]:
    """Threshold detector on smoothed epistemic entropy.

    Args:
        epistemic: Raw per-step epistemic entropy values.
        steps: Corresponding step indices.
        rolling_window: Window size for smoothing.
        baseline_window: ``(start, end)`` step range for baseline stats.
        k_sigma: Number of standard deviations above the mean for the
            threshold.
        min_consecutive: Required consecutive exceedances before firing.

    Returns:
        ``(detect_step, mu, sigma, smoothed)`` where *detect_step* is
        the first step where the detector fires, or ``None``.
    """
    smoothed = rolling_mean(epistemic, rolling_window)
    in_baseline = (steps >= baseline_window[0]) & (steps < baseline_window[1])
    baseline_vals = smoothed[in_baseline]
    mu = float(baseline_vals.mean())
    sigma = float(baseline_vals.std(ddof=1))
    threshold = mu + k_sigma * sigma

    scan_start = int(np.searchsorted(steps, baseline_window[1], side="left"))
    above = smoothed > threshold
    detect_step: int | None = None
    streak = 0
    for i in range(scan_start, len(above)):
        if above[i]:
            streak += 1
            if streak >= min_consecutive:
                detect_step = int(steps[i])
                break
        else:
            streak = 0
    return detect_step, mu, sigma, smoothed


def first_arf_drift_after(
    n_drifts: np.ndarray,
    steps: np.ndarray,
    after: int,
) -> int | None:
    """Return the first step where ``n_drifts_detected`` increments past its value at *after*."""
    pre_idx = np.searchsorted(steps, after, side="right") - 1
    baseline = int(n_drifts[pre_idx]) if pre_idx >= 0 else 0
    mask = (steps >= after) & (n_drifts > baseline)
    if not mask.any():
        return None
    return int(steps[mask][0])
