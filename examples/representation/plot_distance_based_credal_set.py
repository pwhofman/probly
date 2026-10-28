"""============================
Distance-based credal set
============================

A :class:`~probly.representation.credal_set.DistanceBasedCredalSet`
contains every distribution whose total-variation distance to a **nominal**
distribution is at most a given **radius**.  The larger the radius, the wider
the uncertainty around the nominal.

This representation is useful when you have a best-guess distribution but want
to account for a bounded amount of model mis-specification.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from probly.plot import plot_credal_set
from probly.representation.credal_set import (
    DistanceBasedCredalSet,
    create_distance_based_credal_set_from_center_and_radius,
)
from probly.representation.distribution import create_categorical_distribution

# 2 instances over 3 classes, with a shared radius.
nominal = create_categorical_distribution(
    np.array(
        [
            [0.5, 0.3, 0.2],
            [0.2, 0.6, 0.2],
        ]
    ),
)
distance_based: DistanceBasedCredalSet = create_distance_based_credal_set_from_center_and_radius(
    nominal,
    np.array([0.1, 0.1]),
)

print("Shape (batch dims):", distance_based.shape)
print("Nominal:\n", distance_based.nominal)
print("Radius:", distance_based.radius)
print("Lower envelope (max(0, nominal - r)):\n", distance_based.lower())
print("Upper envelope (min(1, nominal + r)):\n", distance_based.upper())

# %%
# On the simplex the feasible region is shown as a filled polygon around the
# nominal (marked with a dot).
plot_credal_set(distance_based, title="Distance-based credal set")
plt.show()
