"""
====================================
Working with samples (`ArraySample`)
====================================

Probly represents repeated stochastic predictions as a "sample". For NumPy-like data,
the concrete implementation is :class:`probly.representation.sampling.sample.ArraySample`.

This example shows:

1) building a sample from repeated model outputs, and
2) summarizing it with ``mean`` and ``std``.
"""

from __future__ import annotations

import numpy as np

from probly.representation.sampling.sample import ArraySample

# Imagine these are 3 stochastic forward passes for 2 instances and 4 classes.
# Shape per pass: (instances, classes)
pass_1 = np.array([[0.1, 0.2, 0.6, 0.1], [0.7, 0.1, 0.1, 0.1]])
pass_2 = np.array([[0.2, 0.2, 0.5, 0.1], [0.6, 0.2, 0.1, 0.1]])
pass_3 = np.array([[0.15, 0.25, 0.5, 0.1], [0.65, 0.15, 0.1, 0.1]])

sample = ArraySample([pass_1, pass_2, pass_3])

mean = sample.mean()
std = sample.std(ddof=0)

print("mean shape:", mean.shape)
print("std shape:", std.shape)
print("mean[0]:", mean[0])
print("std[0]:", std[0])

