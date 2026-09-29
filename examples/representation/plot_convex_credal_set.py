"""========================
Convex credal set
========================

A :class:`~probly.representation.credal_set.ConvexCredalSet`
is defined by the **convex hull** of a set of vertex distributions.  Every
distribution that can be written as a convex combination of the vertices is
considered a member of the set.

This is a common representation in imprecise-probability theory: the vertices
are the extreme points of a credal polytope, and the full set is their convex
closure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from probly.plot import plot_credal_set
from probly.representation.credal_set import ConvexCredalSet, create_convex_credal_set
from probly.representation.distribution import create_categorical_distribution
from probly.representation.sample import create_sample

# 3 vertex distributions over 3 classes, each given for 2 instances.
# Shape per vertex: (instances, classes) = (2, 3)
vertices = [
    create_categorical_distribution(np.array([[0.7, 0.2, 0.1], [0.4, 0.4, 0.2]])),
    create_categorical_distribution(np.array([[0.1, 0.7, 0.2], [0.2, 0.2, 0.6]])),
    create_categorical_distribution(np.array([[0.1, 0.1, 0.8], [0.3, 0.1, 0.6]])),
]
convex: ConvexCredalSet = create_convex_credal_set(create_sample(vertices, sample_axis=0))

print("Shape (batch dims):", convex.shape)
print("Lower envelope (vertex-wise min):\n", convex.lower())
print("Upper envelope (vertex-wise max):\n", convex.upper())

# %%
# On the simplex the convex hull of the vertices is drawn as a filled polygon.
plot_credal_set(convex, title="Convex credal set")
plt.show()
