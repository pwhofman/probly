"""==================================================
Plotting credal sets on a spider (radar) chart
==================================================

The :func:`~probly.plot.plot_credal_set` function automatically switches to a
spider (radar) plot for credal sets with 4 or more classes. Each class
corresponds to a spoke, and probability bounds are rendered as constant-width
bars.

The same five credal set types are supported:

- :class:`~probly.representation.credal_set.array.ArraySingletonCredalSet` --
  a closed envelope connecting point probabilities on each spoke.
- :class:`~probly.representation.credal_set.array.ArrayProbabilityIntervalsCredalSet` --
  constant-width bars on each spoke from lower to upper bound.
- :class:`~probly.representation.credal_set.array.ArrayDistanceBasedCredalSet` --
  the same bars plus a marker at the nominal distribution.
- :class:`~probly.representation.credal_set.array.ArrayConvexCredalSet` --
  member distribution envelopes with an optional min/max envelope.
- :class:`~probly.representation.credal_set.array.ArrayDiscreteCredalSet` --
  individual member distributions as distinct colored lines.

Set ``envelope=True`` to draw a filled envelope connecting the bounds across
spokes. An optional ``ground_truth`` overlay (as an ``ArraySingletonCredalSet``)
can be added to any plot type.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from probly.plot import PlotConfig, plot_credal_set
from probly.representation.credal_set.array import (
    ArrayConvexCredalSet,
    ArrayDiscreteCredalSet,
    ArrayDistanceBasedCredalSet,
    ArrayProbabilityIntervalsCredalSet,
    ArraySingletonCredalSet,
)

NUM_CLASSES = 8
CLASS_LABELS = [f"Class {i}" for i in range(NUM_CLASSES)]

# %%
# Singleton credal set
# --------------------
# A single probability distribution shown as a closed envelope.

singleton = ArraySingletonCredalSet(
    array=np.array([[0.35, 0.20, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03]]),
)
plot_credal_set(singleton, title="Singleton", labels=CLASS_LABELS)
plt.show()

# %%
# Probability intervals
# ---------------------
# Constant-width bars show the per-class probability bounds. By default no
# envelope is drawn (``envelope=False``).

intervals = ArrayProbabilityIntervalsCredalSet(
    lower_bounds=np.array([[0.05, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]]),
    upper_bounds=np.array([[0.80, 0.30, 0.20, 0.15, 0.10, 0.10, 0.08, 0.05]]),
)
plot_credal_set(intervals, title="Probability Intervals", labels=CLASS_LABELS)
plt.show()

# %%
# Probability intervals with envelope
# ------------------------------------
# Setting ``envelope=True`` adds a filled area connecting the bounds across
# spokes. A ground-truth overlay is also shown.

gt = ArraySingletonCredalSet(
    array=np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
)
plot_credal_set(
    intervals,
    title="Probability Intervals (envelope=True)",
    labels=CLASS_LABELS,
    envelope=True,
    ground_truth=gt,
)
plt.show()

# %%
# Distance-based credal set
# -------------------------
# Interval bars derived from a nominal distribution and radius, with a marker
# at the nominal.

distance_based = ArrayDistanceBasedCredalSet(
    nominal=np.array([[0.30, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.05]]),
    radius=0.05,
)
plot_credal_set(distance_based, title="Distance-Based", labels=CLASS_LABELS)
plt.show()

# %%
# Distance-based with envelope
# ----------------------------
# The same credal set with ``envelope=True`` to show the connecting fill.

plot_credal_set(
    distance_based,
    title="Distance-Based (envelope=True)",
    labels=CLASS_LABELS,
    envelope=True,
)
plt.show()

# %%
# Convex credal set
# -----------------
# Multiple vertex distributions drawn as prominent lines. Without
# ``envelope=True`` only the member lines are shown.

convex = ArrayConvexCredalSet(
    array=np.array([
        [
            [0.50, 0.15, 0.10, 0.08, 0.07, 0.05, 0.03, 0.02],
            [0.10, 0.40, 0.15, 0.10, 0.10, 0.05, 0.05, 0.05],
            [0.15, 0.10, 0.35, 0.15, 0.10, 0.05, 0.05, 0.05],
        ],
    ]),
)
plot_credal_set(convex, title="Convex", labels=CLASS_LABELS)
plt.show()

# %%
# Convex with envelope
# --------------------
# Adding ``envelope=True`` fills the min/max envelope between members.

plot_credal_set(
    convex,
    title="Convex (envelope=True)",
    labels=CLASS_LABELS,
    envelope=True,
)
plt.show()

# %%
# Discrete credal set
# -------------------
# Each member distribution is drawn as its own colored line. The ``envelope``
# parameter has no effect on discrete sets.

discrete = ArrayDiscreteCredalSet(
    array=np.array([
        [
            [0.40, 0.20, 0.15, 0.10, 0.08, 0.03, 0.02, 0.02],
            [0.10, 0.35, 0.20, 0.15, 0.08, 0.05, 0.04, 0.03],
            [0.15, 0.10, 0.30, 0.20, 0.10, 0.07, 0.05, 0.03],
        ],
    ]),
)
plot_credal_set(discrete, title="Discrete", labels=CLASS_LABELS)
plt.show()

# %%
# Custom styling
# --------------
# Adjust visual parameters through :class:`~probly.plot.PlotConfig`.

config = PlotConfig(fill_alpha=0.5, line_width=2.5, spider_bar_width=0.08)
plot_credal_set(
    intervals,
    title="Custom Styling",
    labels=CLASS_LABELS,
    config=config,
    envelope=True,
)
plt.show()
