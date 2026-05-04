"""Numpy implementations of sample-based uncertainty measures."""

from __future__ import annotations

import numpy as np

from probly.quantification.measure.sample._common import mean_squared_distance_to_scaled_one_hot
from probly.representation.sample.array import ArraySample


@mean_squared_distance_to_scaled_one_hot.register(ArraySample)
def array_mean_squared_distance_to_scaled_one_hot(sample: ArraySample, scale: float | None = None) -> np.ndarray:
    r"""Numpy impl. Uses :math:`\|h_k - s e_c\|^2 = \|h_k\|^2 - 2s \max_j h_{k,j} + s^2` (no one-hot built)."""
    array = sample.array
    num_classes = array.shape[-1]
    target_scale = float(num_classes) if scale is None else float(scale)

    norm_sq = np.sum(array * array, axis=-1)
    max_logit = np.max(array, axis=-1)
    per_member = norm_sq - 2.0 * target_scale * max_logit + target_scale * target_scale

    if sample.weights is not None:
        return np.average(per_member, axis=sample.sample_axis, weights=sample.weights)
    return np.mean(per_member, axis=sample.sample_axis)
