"""=========================
Sets of outcomes
=========================

A conformal prediction set names the outcomes that stay in play and attaches no
probability to them. ``probly`` stores the classification case as an
:class:`~probly.representation.conformal_set.array.ArrayOneHotConformalSet` and the
regression case as an
:class:`~probly.representation.conformal_set.array.ArrayIntervalConformalSet`.

The uncertainty *is* the size of the set: how many labels had to be kept, or how wide the
interval had to be, to hold the coverage level the calibration step fixed.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from probly.plot import PlotConfig
from probly.representation.conformal_set import (
    create_interval_conformal_set,
    create_onehot_conformal_set,
)

CLASSES = ["cat", "dog", "fox"]
config = PlotConfig()

# The scores the set is built from, one first-order distribution per input.
scores = np.array([[0.05, 0.90, 0.05], [0.47, 0.51, 0.02], [0.34, 0.33, 0.33]])

# The labels kept at the calibrated threshold, one-hot encoded.
# Shape: (instances, classes)
conformal_set = create_onehot_conformal_set(
    np.array([[False, True, False], [True, True, False], [True, True, True]])
)

print("Set membership:\n", conformal_set.array)
print("Set sizes:", conformal_set.set_size)

# %%
# The kept labels carry no odds: inside the set every outcome is simply in play. The
# scores are drawn underneath only to show what the set was built from.
fig, axes = plt.subplots(1, 3, figsize=(8, 2.8), sharey=True)
for index, ax in enumerate(axes):
    colors = [config.categorical_palette[0] if kept else config.color_neutral for kept in conformal_set.array[index]]
    ax.bar(CLASSES, scores[index], color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Set size {conformal_set.set_size[index]}")
axes[0].set_ylabel("Score")
fig.suptitle("Sets of outcomes: kept labels in color, dropped labels in gray")
fig.tight_layout()
plt.show()

# %%
# For regression the same rung is an interval, and its size is the width.
interval_set = create_interval_conformal_set(np.array([2.9, 1.4, 0.2]), np.array([3.5, 5.0, 6.4]))
print("Interval bounds:\n", interval_set.array)
print("Interval widths:", interval_set.set_size)

fig, ax = plt.subplots(figsize=(7, 2.6))
lower, upper = interval_set.array[:, 0], interval_set.array[:, 1]
positions = np.arange(len(lower))
ax.barh(
    positions,
    upper - lower,
    left=lower,
    height=0.4,
    color=config.categorical_palette[0],
)
ax.set_yticks(positions, [f"Input {index}" for index in positions])
ax.set_xlabel("Outcome")
ax.set_title("Regression: an interval per input")
ax.invert_yaxis()
fig.tight_layout()
plt.show()

# %%
# A wide set does not say whether the world or the model made it wide, so the rung is
# silent about the aleatoric/epistemic split. What it buys instead is a coverage
# guarantee that holds whatever the underlying model does, as long as calibration and
# test data are exchangeable.
