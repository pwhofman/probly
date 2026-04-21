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

from river_uncertainty.representation import ARFEnsembleRepresentation, river_arf_to_probly_sample

if TYPE_CHECKING:
    from collections.abc import Iterable

    from river import forest


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
            _record(trace, step, x=x, y=y, arf=arf, rep=rep)

        arf.learn_one(x, y)

    return trace


def _record(
    trace: PrequentialTrace,
    step: int,
    *,
    x: dict[str, float],
    y: object,
    arf: forest.ARFClassifier,
    rep: ARFEnsembleRepresentation,
) -> None:
    entropy_decomp = SecondOrderEntropyDecomposition(rep.sample)
    zero_one_decomp = SecondOrderZeroOneDecomposition(rep.sample)

    bma = rep.bma().probabilities
    y_pred_idx = int(np.argmax(bma))
    y_pred = rep.classes[y_pred_idx]
    max_prob = float(bma[y_pred_idx])

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
    trace.bma_max_prob.append(max_prob)
    trace.n_drifts_detected.append(int(arf.n_drifts_detected()))
