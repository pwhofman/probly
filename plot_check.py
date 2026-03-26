"""Quick sanity-check script for the plotting module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from probly.plot import plot_credal_set
from probly.representation.credal_set.array import (
    ArrayConvexCredalSet,
    ArrayDiscreteCredalSet,
    ArrayDistanceBasedCredalSet,
    ArrayProbabilityIntervalsCredalSet,
    ArraySingletonCredalSet,
)

if TYPE_CHECKING:
    from probly.plot._base import PlotFunction

# ── Tweak data here ──────────────────────────────────────────────────────────

intervals = ArrayProbabilityIntervalsCredalSet(
    lower_bounds=np.array([[0.1, 0.2, 0.3], [0.3, 0.1, 0.1], [0.3, 0.2, 0.1]]),
    upper_bounds=np.array([[0.4, 0.5, 0.6], [0.6, 0.3, 0.7], [0.9, 1, 0.1]]),
)

distance_based = ArrayDistanceBasedCredalSet(
    nominal=np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2]]),
    radius=0.1,
)

convex = ArrayConvexCredalSet(
    array=np.array(
        [
            [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.1, 0.1, 0.8]],
            [[0.5, 0.4, 0.1], [0.3, 0.3, 0.4], [0.4, 0.2, 0.4]],
        ]
    ),
)

discrete = ArrayDiscreteCredalSet(
    array=np.array(
        [
            [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]],
            [[0.4, 0.4, 0.2], [0.3, 0.2, 0.5]],
        ]
    ),
)

singleton_batched = ArraySingletonCredalSet(
    array=np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]]),
)

singleton_scalar = ArraySingletonCredalSet(array=np.array([0.4, 0.4, 0.2]))

# ── Plot all ─────────────────────────────────────────────────────────────────

EXAMPLES = {
    "ProbabilityIntervals": intervals,
    "DistanceBased": distance_based,
    "Convex": convex,
    "Discrete": discrete,
    "Singleton (batched)": singleton_batched,
    "Singleton (scalar)": singleton_scalar,
}

if __name__ == "__main__":
    for title, data in EXAMPLES.items():
        plot_credal_set(data, title=title, gridlines=True)
        plt.show()

    a: PlotFunction = plot_credal_set
