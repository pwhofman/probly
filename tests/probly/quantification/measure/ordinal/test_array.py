"""Tests for ordinal-variance / ordinal-entropy uncertainty measures (numpy)."""

from __future__ import annotations

import numpy as np


class TestArrayOrdinal:
    """Numpy implementations of ordinal-variance / ordinal-entropy."""

    def test_variance_on_distribution(self) -> None:
        from probly.quantification.measure.ordinal import ordinal_variance  # noqa: PLC0415
        from probly.representation.distribution.array_categorical import (  # noqa: PLC0415
            ArrayProbabilityCategoricalDistribution,
        )

        d = ArrayProbabilityCategoricalDistribution(array=np.array([[0.5, 0.5, 0.0]]))
        out = ordinal_variance(d)
        # cdf = [0.5, 1.0] -> excluding last bin -> [0.5]
        # variance = 0.5 * (1 - 0.5) = 0.25
        np.testing.assert_allclose(out, [0.25])

    def test_variance_on_raw_array_branch(self) -> None:
        # Direct call to the registered handler with a raw array exercises the
        # else branch of the dispatch.
        from probly.quantification.measure.ordinal.array import (  # noqa: PLC0415
            array_categorical_ordinal_variance,
        )

        out = array_categorical_ordinal_variance(np.array([[0.5, 0.5]]))
        np.testing.assert_allclose(out, [0.25])

    def test_entropy_on_distribution(self) -> None:
        from probly.quantification.measure.ordinal import ordinal_entropy  # noqa: PLC0415
        from probly.representation.distribution.array_categorical import (  # noqa: PLC0415
            ArrayProbabilityCategoricalDistribution,
        )

        # For a uniform 2-class distribution, the binary entropy is log(2) (in nats).
        d = ArrayProbabilityCategoricalDistribution(array=np.array([[0.5, 0.5]]))
        out = ordinal_entropy(d)
        np.testing.assert_allclose(out, [np.log(2)], atol=1e-6)

    def test_entropy_normalized_base(self) -> None:
        from probly.quantification.measure.ordinal import ordinal_entropy  # noqa: PLC0415
        from probly.representation.distribution.array_categorical import (  # noqa: PLC0415
            ArrayProbabilityCategoricalDistribution,
        )

        # Uniform 2-class, normalized base -> entropy = 1.
        d = ArrayProbabilityCategoricalDistribution(array=np.array([[0.5, 0.5]]))
        out = ordinal_entropy(d, base="normalize")
        np.testing.assert_allclose(out, [1.0], atol=1e-6)
