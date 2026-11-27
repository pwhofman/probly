"""NumPy implementations for generic credal-set representers."""

from __future__ import annotations

import numpy as np

from probly.representation.distribution.numpy_categorical import (
    NumpyCategoricalDistributionSample,
    NumpyProbabilityCategoricalDistribution,
)
from probly.representer.credal_set import compute_representative_sample


@compute_representative_sample.register(NumpyCategoricalDistributionSample)
def numpy_compute_representative_sample(
    sample: NumpyCategoricalDistributionSample,
    alpha: float,
    distance: str,
) -> NumpyCategoricalDistributionSample:
    """Select predictions closest to the sample mean under the requested distance."""
    if distance == "euclidean":
        if alpha == 0.0:
            return sample
        probs = sample.samples.probabilities  # (M, ..., C)
        mean = np.mean(probs, axis=0, keepdims=True)
        dist = np.linalg.norm(probs - mean, axis=-1)  # (M, ...)
        k = int(sample.sample_size * (1 - alpha))
        k = max(k, 1)
        idx = np.argsort(dist, axis=0)[:k]
        selected_probs = np.take_along_axis(probs, idx[..., np.newaxis], axis=0)
        selected = NumpyCategoricalDistributionSample(
            array=NumpyProbabilityCategoricalDistribution(selected_probs),
            sample_axis=0,
        )
        return selected.move_sample_axis(sample.sample_axis)  # ty:ignore[invalid-return-type]
    msg = f"Distance {distance} not implemented for NumPy arrays."
    raise NotImplementedError(msg)
