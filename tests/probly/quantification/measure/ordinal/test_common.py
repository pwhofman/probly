"""Tests for the ordinal-measure flexdispatch fallbacks."""

from __future__ import annotations

import pytest


class TestQuantificationOrdinalFallbacks:
    """Each ordinal-measure flexdispatch raises NotImplementedError without a registered handler."""

    def _candidates(self) -> list[str]:
        return [
            "ordinal_variance",
            "ordinal_entropy",
            "ordinal_variance_of_expected_predictive_distribution",
            "ordinal_conditional_variance",
            "ordinal_mutual_information_variance",
            "ordinal_entropy_of_expected_predictive_distribution",
            "ordinal_mutual_information_entropy",
            "ordinal_conditional_entropy",
            "categorical_variance_total",
            "categorical_variance_aleatoric",
            "labelwise_entropy",
            "labelwise_variance",
            "labelwise_entropy_of_expected_predictive_distribution",
            "labelwise_conditional_entropy",
        ]

    def test_each_ordinal_measure_raises_for_unknown_input(self) -> None:
        from probly.quantification.measure.ordinal import _common as ordinal  # noqa: PLC0415

        # Use a sentinel object that no handler is registered for.
        for name in self._candidates():
            func = getattr(ordinal, name)
            with pytest.raises(NotImplementedError):
                func(object())
