"""========================
Discrete credal set
========================

An :class:`~probly.representation.credal_set.array.ArrayDiscreteCredalSet`
represents a **finite set** of candidate probability distributions.  Unlike a
convex credal set, only the listed members are considered part of the set — no
interpolation between them is implied.

This is the natural representation when you have a small collection of model
predictions (e.g. from an ensemble) and want to keep each member separate.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from probly.plot import plot_credal_set
from probly.representation.credal_set.array import ArrayDiscreteCredalSet

# 2 instances, each with 3 member distributions over 3 classes.
# Shape: (instances, members, classes) = (2, 3, 3)
discrete = ArrayDiscreteCredalSet(
    array=np.array(
        [
            [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.4, 0.2, 0.4]],
            [[0.1, 0.6, 0.3], [0.3, 0.4, 0.3], [0.2, 0.5, 0.3]],
        ]
    ),
)

print("Shape (batch dims):", discrete.shape)
print("Lower envelope (element-wise min across members):\n", discrete.lower())
print("Upper envelope (element-wise max across members):\n", discrete.upper())

# %%
# On the simplex each member is shown as an individual scatter marker.
plot_credal_set(discrete, title="Discrete credal set")
plt.show()
