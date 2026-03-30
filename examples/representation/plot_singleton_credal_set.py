"""========================
Singleton credal set
========================

A :class:`~probly.representation.credal_set.array.ArraySingletonCredalSet`
contains exactly **one** probability distribution per instance — there is no
epistemic uncertainty.  Because the set has a single member, both the lower
and upper envelopes are identical to that distribution.

This is the simplest credal set and is useful as a baseline or when a model
produces a single point prediction.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from probly.plot import plot_credal_set
from probly.representation.credal_set.array import ArraySingletonCredalSet

# A batch of 3 instances over 3 classes, each with one precise distribution.
singleton = ArraySingletonCredalSet(
    array=np.array(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.2, 0.7],
        ]
    ),
)

print("Shape (batch dims):", singleton.shape)
print("Lower envelope:\n", singleton.lower())
print("Upper envelope:\n", singleton.upper())
print("Lower == Upper:", np.allclose(singleton.lower(), singleton.upper()))

# %%
# On the simplex each singleton is a single point.
plot_credal_set(singleton, title="Singleton credal set")
plt.show()
