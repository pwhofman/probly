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


intervals = ArrayProbabilityIntervalsCredalSet(
    lower_bounds=np.array([[0.1, 0.2, 0.3], [0.3, 0.1, 0.1], [0.1, 0.2, 0.1]]),
    upper_bounds=np.array([[0.4, 0.5, 0.6], [0.6, 0.3, 0.7], [0.1, 1.0, 0.1]]),
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

EXAMPLES = {
    "ProbabilityIntervals": intervals,
    "DistanceBased": distance_based,
    "Convex": convex,
    "Discrete": discrete,
    "Singleton (batched)": singleton_batched,
    "Singleton (scalar)": singleton_scalar,
}

# --- Binary (2-class) examples ---

binary_intervals = ArrayProbabilityIntervalsCredalSet(
    lower_bounds=np.array([[0.2, 0.8], [0.4, 0.6]]),
    upper_bounds=np.array([[0.5, 0.5], [0.7, 0.3]]),
)

binary_intervals_zero_area = ArrayProbabilityIntervalsCredalSet(
    lower_bounds=np.array([[0.3, 0.7]]),
    upper_bounds=np.array([[0.3, 0.7]]),
)

binary_distance_based = ArrayDistanceBasedCredalSet(
    nominal=np.array([[0.4, 0.6]]),
    radius=0.15,
)

binary_singleton = ArraySingletonCredalSet(
    array=np.array([[0.3, 0.7], [0.6, 0.4]]),
)

binary_discrete = ArrayDiscreteCredalSet(
    array=np.array([[[0.2, 0.8], [0.5, 0.5], [0.7, 0.3]]]),
)

binary_convex = ArrayConvexCredalSet(
    array=np.array([[[0.1, 0.9], [0.4, 0.6], [0.8, 0.2]]]),
)

BINARY_EXAMPLES = {
    "Binary ProbabilityIntervals": binary_intervals,
    "Binary ProbabilityIntervals (zero-area)": binary_intervals_zero_area,
    "Binary DistanceBased": binary_distance_based,
    "Binary Singleton": binary_singleton,
    "Binary Discrete": binary_discrete,
    "Binary Convex": binary_convex,
}

if __name__ == "__main__":
    for title, data in BINARY_EXAMPLES.items():
        plot_credal_set(data, title=title)
        plt.show()

    for title, data in EXAMPLES.items():
        plot_credal_set(data, title=title, gridlines=True)
        plt.show()

    a: PlotFunction = plot_credal_set
