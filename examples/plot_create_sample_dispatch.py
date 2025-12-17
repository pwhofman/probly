"""
=========================================
Automatic sample construction (dispatcher)
=========================================

You typically don't want to care about the concrete sample type. ``probly`` provides
``create_sample`` which selects the best representation based on the sample element type.

For example:

- lists of NumPy arrays become an :class:`~probly.representation.sampling.sample.ArraySample`
- lists of Python scalars become an :class:`~probly.representation.sampling.sample.ArraySample`
- other objects fall back to :class:`~probly.representation.sampling.sample.ListSample`
"""

from __future__ import annotations

import numpy as np

from probly.representation.sampling.sample import create_sample

samples = [
    np.array([[0.2, 0.8], [0.7, 0.3]]),
    np.array([[0.1, 0.9], [0.6, 0.4]]),
    np.array([[0.3, 0.7], [0.8, 0.2]]),
]

sample = create_sample(samples)

print("sample type:", type(sample).__name__)
print("mean:", sample.mean())
