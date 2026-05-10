"""Tests for the distribution-measure flexdispatch fallbacks."""

from __future__ import annotations

import pytest


class TestDistributionMeasureFallbacks:
    """Each top-level distribution-measure dispatch raises NotImplementedError."""

    def test_conditional_entropy_raises(self) -> None:
        from probly.quantification.measure.distribution._common import conditional_entropy  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="Conditional entropy"):
            conditional_entropy(object())

    def test_mutual_information_raises(self) -> None:
        from probly.quantification.measure.distribution._common import mutual_information  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="Mutual information"):
            mutual_information(object())

    def test_max_probability_complement_of_expected_raises(self) -> None:
        from probly.quantification.measure.distribution._common import (  # noqa: PLC0415
            max_probability_complement_of_expected,
        )

        with pytest.raises(NotImplementedError, match="Max probability complement"):
            max_probability_complement_of_expected(object())

    def test_expected_max_probability_complement_raises(self) -> None:
        from probly.quantification.measure.distribution._common import (  # noqa: PLC0415
            expected_max_probability_complement,
        )

        with pytest.raises(NotImplementedError, match="Expected max probability"):
            expected_max_probability_complement(object())

    def test_max_disagreement_raises(self) -> None:
        from probly.quantification.measure.distribution._common import max_disagreement  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="Max disagreement"):
            max_disagreement(object())

    def test_vacuity_raises(self) -> None:
        from probly.quantification.measure.distribution._common import vacuity  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="Vacuity"):
            vacuity(object())

    def test_dempster_shafer_raises(self) -> None:
        from probly.quantification.measure.distribution._common import dempster_shafer_uncertainty  # noqa: PLC0415

        with pytest.raises(NotImplementedError, match="Dempster-Shafer"):
            dempster_shafer_uncertainty(object())


class TestQuantificationDistributionFallbacks:
    """`_common.py` flexdispatch fallbacks for distribution measures."""

    def test_each_distribution_measure_raises_for_unknown_input(self) -> None:
        from probly.quantification.measure.distribution import _common as distm  # noqa: PLC0415

        candidates = [
            ("variance", NotImplementedError),
            ("entropy", NotImplementedError),
            ("variance_of_expected_predictive_distribution", NotImplementedError),
            ("conditional_variance", NotImplementedError),
            ("mutual_information_variance", NotImplementedError),
            ("entropy_of_expected_predictive_distribution", NotImplementedError),
            ("mutual_information_entropy", NotImplementedError),
            ("conditional_entropy", NotImplementedError),
        ]
        for name, exc in candidates:
            func = getattr(distm, name, None)
            if func is None:
                continue
            with pytest.raises(exc):
                func(object())
