"""===============================================
Point predictions and first-order distributions
===============================================

The bottom two rungs of the representation ladder, on the same three-class problem
("cat", "dog", "fox"):

1. a **point prediction**, one outcome and nothing else, and
2. a **first-order distribution**, one probability distribution over the outcomes, here an
   :class:`~probly.representation.distribution.array_categorical.ArrayProbabilityCategoricalDistribution`.

The two distributions below have the same ``argmax``, so they collapse to the same point
prediction. What separates them is only visible one rung up.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from probly.plot import PlotConfig
from probly.representation.distribution import (
    ArrayGaussianDistribution,
    ArrayProbabilityCategoricalDistribution,
)

CLASSES = ["cat", "dog", "fox"]
config = PlotConfig()

# Two predictions, one per input. Shape: (instances, classes) = (2, 3)
distribution = ArrayProbabilityCategoricalDistribution(
    array=np.array([[0.49, 0.51, 0.00], [0.33, 0.34, 0.33]]),
)

print("Shape (batch dims):", distribution.shape)
print("Number of classes:", distribution.num_classes)
print("Probabilities:\n", distribution.probabilities)

# %%
# The point prediction keeps the ``argmax`` and discards everything else. Both inputs
# land on the same label, so at this rung the two predictions are indistinguishable.
point_prediction = np.argmax(distribution.probabilities, axis=-1)
print("Point predictions:", [CLASSES[index] for index in point_prediction])

fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
for index, ax in enumerate(axes):
    heights = np.zeros(len(CLASSES))
    heights[point_prediction[index]] = 1.0
    ax.bar(CLASSES, heights, color=config.color_neutral)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Input {index}")
axes[0].set_ylabel("Point prediction")
fig.suptitle("Zeroth order: one outcome, no scale")
fig.tight_layout()

# %%
# The first-order distribution keeps the odds. Input 0 has two outcomes competing and a
# third ruled out, input 1 has all three competing -- a difference the rung below has no
# slot for.
fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
for index, ax in enumerate(axes):
    ax.bar(CLASSES, distribution.probabilities[index], color=config.categorical_palette[0])
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Input {index}")
axes[0].set_ylabel("Probability")
fig.suptitle("First order: a distribution over outcomes")
fig.tight_layout()

# %%
# For regression the same rung is a distribution over the real line, for instance an
# :class:`~probly.representation.distribution.array_gaussian.ArrayGaussianDistribution`
# holding a mean and a variance per instance.
gaussian = ArrayGaussianDistribution(mean=np.array([3.2, 3.2]), var=np.array([0.05, 0.9]))
print("Means:", gaussian.mean)
print("Standard deviations:", gaussian.std)

grid = np.linspace(0.0, 6.5, 400)
fig, ax = plt.subplots(figsize=(7, 2.8))
for index in range(2):
    ax.plot(
        grid,
        norm.pdf(grid, gaussian.mean[index], gaussian.std[index]),
        color=config.categorical_palette[index],
        label=f"Input {index}: var = {gaussian.var[index]:.2f}",
    )
ax.set_xlabel("Outcome")
ax.set_ylabel("Density")
ax.set_title("Regression: same mean, different spread")
ax.legend()
fig.tight_layout()
plt.show()
