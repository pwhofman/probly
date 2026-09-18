"""Tests for sample-based numpy uncertainty measures."""

from __future__ import annotations

import numpy as np


class TestSampleMeasureArray:
    """Sample-based numpy uncertainty measures."""

    def test_mean_squared_distance_array(self) -> None:
        from probly.quantification.measure.sample import (  # noqa: PLC0415
            mean_squared_distance_to_scaled_one_hot,
        )
        from probly.representation.distribution.array_categorical import (  # noqa: PLC0415
            ArrayCategoricalDistributionSample,
            ArrayProbabilityCategoricalDistribution,
        )

        probs = np.array([[[0.6, 0.3, 0.1]], [[0.1, 0.6, 0.3]]])
        dist = ArrayProbabilityCategoricalDistribution(array=probs)
        sample = ArrayCategoricalDistributionSample(array=dist, sample_axis=0)
        result = mean_squared_distance_to_scaled_one_hot(sample, scale=1.0)
        assert result.shape == (1,)
        assert np.isfinite(result).all()

    def test_total_logit_sample_variance_array(self) -> None:
        from probly.quantification.measure.sample import total_logit_sample_variance  # noqa: PLC0415
        from probly.representation.distribution.array_categorical import (  # noqa: PLC0415
            ArrayCategoricalDistributionSample,
            ArrayProbabilityCategoricalDistribution,
        )

        probs = np.array([[[0.6, 0.3, 0.1]], [[0.1, 0.6, 0.3]]])
        dist = ArrayProbabilityCategoricalDistribution(array=probs)
        sample = ArrayCategoricalDistributionSample(array=dist, sample_axis=0)
        result = total_logit_sample_variance(sample)
        assert result.shape == (1,)
        assert np.isfinite(result).all()
