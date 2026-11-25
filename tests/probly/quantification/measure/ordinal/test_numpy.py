"""Tests for ordinal-variance / ordinal-entropy uncertainty measures (numpy)."""

from __future__ import annotations

import numpy as np


class TestArrayOrdinal:
    """Numpy implementations of ordinal-variance / ordinal-entropy."""

    def test_variance_on_distribution(self) -> None:
        from probly.quantification.measure.ordinal import ordinal_variance  # noqa: PLC0415
        from probly.representation.distribution.numpy_categorical import (  # noqa: PLC0415
            NumpyProbabilityCategoricalDistribution,
        )

        d = NumpyProbabilityCategoricalDistribution(array=np.array([[0.5, 0.5, 0.0]]))
        out = ordinal_variance(d)
        # cdf = [0.5, 1.0] -> excluding last bin -> [0.5]
        # variance = 0.5 * (1 - 0.5) = 0.25
        np.testing.assert_allclose(out, [0.25])

    def test_entropy_on_distribution(self) -> None:
        from probly.quantification.measure.ordinal import ordinal_entropy  # noqa: PLC0415
        from probly.representation.distribution.numpy_categorical import (  # noqa: PLC0415
            NumpyProbabilityCategoricalDistribution,
        )

        # For a uniform 2-class distribution, the binary entropy is log(2) (in nats).
        d = NumpyProbabilityCategoricalDistribution(array=np.array([[0.5, 0.5]]))
        out = ordinal_entropy(d)
        np.testing.assert_allclose(out, [np.log(2)], atol=1e-6)

    def test_entropy_normalized_base(self) -> None:
        from probly.quantification.measure.ordinal import ordinal_entropy  # noqa: PLC0415
        from probly.representation.distribution.numpy_categorical import (  # noqa: PLC0415
            NumpyProbabilityCategoricalDistribution,
        )

        # Uniform 2-class, normalized base -> entropy = 1.
        d = NumpyProbabilityCategoricalDistribution(array=np.array([[0.5, 0.5]]))
        out = ordinal_entropy(d, base="normalize")
        np.testing.assert_allclose(out, [1.0], atol=1e-6)

    def test_entropy_is_finite_when_a_cumulative_probability_rounds_above_one(self) -> None:
        from probly.quantification.measure.ordinal import ordinal_entropy  # noqa: PLC0415
        from probly.representation.distribution.numpy_categorical import (  # noqa: PLC0415
            NumpyProbabilityCategoricalDistribution,
        )

        # Ten classes of 0.1 accumulate to 1.0000001 in float32 before the empty last class.
        d = NumpyProbabilityCategoricalDistribution(array=np.array([[0.1] * 10 + [0.0]], dtype=np.float32))
        out = ordinal_entropy(d)
        splits = np.arange(1, 10) / 10
        expected = -(splits * np.log(splits) + (1 - splits) * np.log(1 - splits)).sum()
        np.testing.assert_allclose(out, [expected], rtol=1e-5)
