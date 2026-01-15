"""Working with samples (`ArraySample`).

``probly`` represents repeated stochastic predictions as a "sample". For NumPy-like data,
the concrete implementation is :class:`probly.representation.sampling.sample.ArraySample`.

This example shows:

1) building a sample from repeated model outputs, and
2) summarizing it with ``mean`` and ``std``, and
3) visualizing predictive uncertainty as an error-bar plot.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from probly.representation.sampling.sample import ArraySample

# Imagine these are 3 stochastic forward passes for 2 instances and 4 classes.
# Shape per pass: (instances, classes)
pass_1 = np.array([[0.1, 0.2, 0.6, 0.1], [0.7, 0.1, 0.1, 0.1]])
pass_2 = np.array([[0.2, 0.2, 0.5, 0.1], [0.6, 0.2, 0.1, 0.1]])
pass_3 = np.array([[0.15, 0.25, 0.5, 0.1], [0.65, 0.15, 0.1, 0.1]])

data_stack = np.array([pass_1, pass_2, pass_3])


sample = ArraySample(data_stack, sample_axis=0)

mean = np.mean(data_stack, axis=0)
std = np.std(data_stack, axis=0, ddof=0)

print("mean shape:", mean.shape)
print("std shape:", std.shape)
print("mean[0]:", mean[0])
print("std[0]:", std[0])

# Plot mean ± std for instance 0
classes = np.arange(mean.shape[1])
plt.figure(figsize=(6, 3))
plt.errorbar(classes, mean[0], yerr=std[0], fmt="o", capsize=4)
plt.xticks(classes)
plt.ylim(0, 1)
plt.xlabel("Class index")
plt.ylabel("Probability")
plt.title("Instance 0: mean ± std across stochastic passes")
plt.tight_layout()
